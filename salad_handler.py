from fastapi import FastAPI, Body, HTTPException
import os
import time
import requests
import base64
import traceback
import runpod
import uvicorn
from requests.adapters import HTTPAdapter, Retry
from huggingface_hub import HfApi

app = FastAPI()

BASE_URI = 'http://127.0.0.1:3000'
TIMEOUT = 600
POST_RETRIES = 3

session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
machine_id = os.environ.get('SALAD_MACHINE_ID', '')

# ---------------------------------------------------------------------------- #
#                               Utility Functions                              #
# ---------------------------------------------------------------------------- #

def wait_for_service(url):
    retries = 0

    while True:
        try:
            requests.get(url)
            return
        except requests.exceptions.RequestException:
            retries += 1

            # Only log every 15 retries so the logs don't get spammed
            if retries % 15 == 0:
                print('INFO: Service not ready yet. Retrying...')
        except Exception as err:
            print(f'ERROR: {err}')

        time.sleep(0.2)


def send_get_request(endpoint):
    return session.get(
        url=f'{BASE_URI}/{endpoint}',
        timeout=TIMEOUT
    )


def send_post_request(endpoint, payload, retry=0):
    response = session.post(
        url=f'{BASE_URI}/{endpoint}',
        json=payload,
        timeout=TIMEOUT
    )

    # Retry the post request in case the model has not completed loading yet
    if response.status_code == 404:
        if retry < POST_RETRIES:
            retry += 1
            print(f'WARNING: Received HTTP 404 from endpoint: {endpoint}. Retrying: {retry}')
            time.sleep(0.2)
            send_post_request(endpoint, payload, retry)

    return response

def download(inp):
    source_url = inp['payload']['source_url']
    download_path = inp['payload']['download_path']
    process_id = os.getpid()
    temp_path = f"{download_path}.{process_id}"

    # Download the file and save it as a temporary file
    with requests.get(source_url, stream=True) as r:
        r.raise_for_status()
        with open(temp_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # Rename the temporary file to the actual file name
    os.rename(temp_path, download_path)
    print(f'INFO: {source_url} successfully downloaded to {download_path}')

    return {
        'msg': 'Download successful',
        'source_url': source_url,
        'download_path': download_path
    }


def sync(inp):
    repo_id = inp['payload']['repo_id']
    sync_path = inp['payload']['sync_path']
    hf_token = inp['payload']['hf_token']

    api = HfApi()

    models = api.list_repo_files(
        repo_id=repo_id,
        token=hf_token
    )

    synced_count = 0
    synced_files = []

    for model in models:
        folder = os.path.dirname(model)
        dest_path = f'{sync_path}/{model}'

        if folder and not os.path.exists(dest_path):
            print(f'Syncing {model} to {dest_path}')

            uri = api.hf_hub_download(
                token=hf_token,
                repo_id=repo_id,
                filename=model,
                local_dir=sync_path,
                local_dir_use_symlinks=False
            )

            if uri:
                synced_count += 1
                synced_files.append(dest_path)

    return {
        'synced_count': synced_count,
        'synced_files': synced_files
    }

def is_url(s):
    return s.startswith('http://') or s.startswith('https://')

def convert_image_to_base64(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure that the request was successful
    return base64.b64encode(response.content).decode('utf-8')

def process_image_fields(payload):
    if 'init_images' in payload:
        payload['init_images'] = [
            convert_image_to_base64(image) if is_url(image) else image
            for image in payload['init_images']
        ]
    
    if 'mask' in payload and is_url(payload['mask']):
        payload['mask'] = convert_image_to_base64(payload['mask'])

    if 'alwayson_scripts' in payload:
        if 'reactor' in payload['alwayson_scripts']:
            first_arg = payload['alwayson_scripts']['reactor']['args'][0]
            if is_url(first_arg):
                payload['alwayson_scripts']['reactor']['args'][0] = convert_image_to_base64(first_arg)

        if 'controlnet' in payload['alwayson_scripts']:
            image = payload['alwayson_scripts']['controlnet']['args'][0]['image']
            if is_url(image):
                payload['alwayson_scripts']['controlnet']['args'][0]['image'] = convert_image_to_base64(image)

def reallocate_machine():
    organization_name = os.environ.get('ORGANIZATION_NAME')
    project_name = os.environ.get('PROJECT_NAME')
    container_group_name = os.environ.get('CONTAINER_GROUP_NAME')
    salad_api_key = os.environ.get('SALAD_API_KEY')
    url = f"https://api.salad.com/api/public/organizations/{organization_name}/projects/{project_name}/containers/{container_group_name}/instances/{machine_id}/reallocate"
    headers = {"Salad-Api-Key": salad_api_key}
    try:
        reallocate_response = requests.post(url, headers=headers)
        print(f"Reallocate response status code: {reallocate_response.status_code}")
    except requests.exceptions.RequestException as req_e:
        print(f"ERROR: Failed to reallocate after connection or runtime error: {req_e}")

# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #

@app.post("/api")
def handler(inp: dict = Body(...)):
    if inp.get("probe") is True:
        return {"probe": "ok"}

    endpoint = inp['api']['endpoint']
    method = inp['api']['method']
    payload = inp['payload']

    process_image_fields(payload)

    try:
        print(f'INFO: Sending {method} request to: /{endpoint}')

        if endpoint == 'v1/download':
            return download(inp)
        elif endpoint == 'v1/sync':
            return sync(inp)
        elif method == 'GET':
            response = send_get_request(endpoint)
        elif method == 'POST':
            response = send_post_request(endpoint, payload)
        
        response_json = response.json()

        if response.status_code == 200:
            return response_json
        elif 'error' in response_json and response_json['error'] == 'RuntimeError':
            reallocate_machine()
            raise HTTPException(status_code=503, detail="Runtime error detected, reallocating resources.")
        else:
            print(f'ERROR: HTTP Status code: {response.status_code}. Machine ID: {machine_id}')
            print(f'ERROR: Response: {response_json}')

            return {
                'error': f'A1111 status code: {response.status_code}',
                'output': response_json
            }
    except requests.exceptions.ConnectionError as e:
        print(f"ERROR: Connection error occurred: {e}")
        reallocate_machine()
        raise HTTPException(status_code=503, detail="Connection error occurred.")
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f'ERROR: An exception was raised on machine {machine_id}: {e}\n{error_trace}')
        raise HTTPException(status_code=500, detail=f"An internal server error occurred on machine on machine {machine_id}:\n{error_trace}")

if __name__ == "__main__":
    wait_for_service(f'{BASE_URI}/sdapi/v1/sd-models')
    print('INFO: Automatic1111 API is ready')
    uvicorn.run("salad_handler:app", host="0.0.0.0", port=80, log_level="error")
