import os
import json
import shutil
from os.path import join, dirname, abspath, exists
from os import makedirs, symlink, readlink
import platform
import subprocess
import sys
import importlib.util
import re
import torch
import cpuinfo
import packaging.tags
from requests import get
import asyncio
import inspect
import aiohttp
from server import PromptServer
from tqdm import tqdm
import pkg_resources



def get_python_version():
    """Return the Python version in a concise format, e.g., '39' for Python 3.9."""
    version_match = re.match(r"3\.(\d+)", platform.python_version())
    if version_match:
        return "3" + version_match.group(1)
    else:
        return None

def get_system_info():
    """Gather system information related to NVIDIA GPU, CUDA version, AVX2 support, Python version, OS, and platform tag."""
    system_info = {
        'gpu': False,
        'cuda_version': None,
        'avx2': False,
        'python_version': get_python_version(),
        'os': platform.system(),
        'os_bit': platform.architecture()[0].replace("bit", ""),
        'platform_tag': None,
    }

    # Check for NVIDIA GPU and CUDA version
    if importlib.util.find_spec('torch'): 
        system_info['gpu'] = torch.cuda.is_available()
        if system_info['gpu']:
            system_info['cuda_version'] = "cu" + torch.version.cuda.replace(".", "").strip()
    
    # Check for AVX2 support
    if importlib.util.find_spec('cpuinfo'):
        try:
            # Attempt to import the cpuinfo module
            import cpuinfo
            
            # Safely attempt to retrieve CPU flags
            cpu_info = cpuinfo.get_cpu_info()
            if cpu_info and 'flags' in cpu_info:
                # Check if 'avx2' is among the CPU flags
                system_info['avx2'] = 'avx2' in cpu_info['flags']
            else:
                # Handle the case where CPU info is unavailable or does not contain 'flags'
                system_info['avx2'] = False
        except Exception as e:
            # Handle unexpected errors gracefully
            print(f"Error retrieving CPU information: {e}")
            system_info['avx2'] = False
    else:
        # Handle the case where the cpuinfo module is not installed
        print("cpuinfo module not available.")
        system_info['avx2'] = False
    # Determine the platform tag
    if importlib.util.find_spec('packaging.tags'):        
        system_info['platform_tag'] = next(packaging.tags.sys_tags()).platform

    return system_info

def latest_lamacpp():
    try:        
        response = get("https://api.github.com/repos/abetlen/llama-cpp-python/releases/latest")
        return response.json()["tag_name"].replace("v", "")
    except Exception:
        return "0.2.20"

def install_package(package_name, custom_command=None):
    if not package_is_installed(package_name):
        print(f"Installing {package_name}...")
        command = [sys.executable, "-m", "pip", "install", package_name, "--no-cache-dir"]
        if custom_command:
            command += custom_command.split()
        subprocess.check_call(command)
    else:
        print(f"{package_name} is already installed.")

def package_is_installed(package_name):
    return importlib.util.find_spec(package_name) is not None

def install_llama(system_info):
    imported = package_is_installed("llama-cpp-python") or package_is_installed("llama_cpp")
    if imported:
        print("llama-cpp installed")
    else:
        lcpp_version = latest_lamacpp()
        base_url = "https://github.com/abetlen/llama-cpp-python/releases/download/v"
        avx = "AVX2" if system_info['avx2'] else "AVX"
        if system_info['gpu']:
            cuda_version = system_info['cuda_version']
            custom_command = f"--force-reinstall --no-deps --index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/{avx}/{cuda_version}"
        elif "arm64" in system_info['platform_tag']:
            custom_command = None
        else:
            custom_command = f"{base_url}{lcpp_version}/llama_cpp_python-{lcpp_version}-{system_info['platform_tag']}.whl"
        install_package("llama-cpp-python", custom_command=custom_command)

def install_autogptq(system_info):
    # Check OS compatibility
    imported = package_is_installed("auto_gptq")
    if imported:
        print("AutoGPTQ installed")
    else:
        if system_info['os'] not in ['Linux', 'Windows']:
            print("AutoGPTQ is not supported on your operating system.")
            return
        
        # Prepare base install command
        base_command = [sys.executable, "-m", "pip", "install", "auto-gptq"]
        
        # Determine the specific install command based on GPU and CUDA/ROCm version
        if system_info['gpu']:
            if 'cuda_version' in system_info and system_info['cuda_version'] in ['cu118', 'cu121']:
                if system_info['cuda_version'] == 'cu118':
                    base_command += ["--extra-index-url", "https://huggingface.github.io/autogptq-index/whl/cu118/"]
                # No extra URL needed for cu121 as it's the default
            elif 'rocm_version' in system_info and system_info['rocm_version'] == 'rocm573':
                base_command += ["--extra-index-url", "https://huggingface.github.io/autogptq-index/whl/rocm573/"]
            else:
                print("Unsupported GPU configuration for AutoGPTQ.")
                return
        else:
            print("No GPU detected. AutoGPTQ installation requires a GPU with CUDA or ROCm support.")
            return
        
        # Execute the installation command
        try:
            print(f"Installing AutoGPTQ with command: {' '.join(base_command)}")
            subprocess.check_call(base_command)
        except Exception as e:
            print(f"Failed to install AutoGPTQ: {e}")

config = None

def is_logging_enabled():
    config = get_extension_config()
    if "logging" not in config:
        return False
    return config["logging"]

def log(message, type=None, always=False, name=None):
    if not always and not is_logging_enabled():
        return

    if type is not None:
        message = f"[{type}] {message}"

    if name is None:
        name = get_extension_config()["name"]

    print(f"(vlmnodes:{name}) {message}")

def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def get_comfy_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(inspect.getfile(PromptServer))
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def get_web_ext_dir():
    config = get_extension_config()
    name = config["name"]
    dir = get_comfy_dir("web/extensions/vlmnodes")
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = os.path.join(dir, name)
    return dir


def get_extension_config(reload=False):
    global config
    if reload == False and config is not None:
        return config

    config_path = get_ext_dir("vlmnodes.json")
    default_config_path = get_ext_dir("vlmnodes.default.json")
    if not os.path.exists(config_path):
        if os.path.exists(default_config_path):
            shutil.copy(default_config_path, config_path)
            if not os.path.exists(config_path):
                log(f"Failed to create config at {config_path}", type="ERROR", always=True, name="???")
                print(f"Extension path: {get_ext_dir()}")
                return {"name": "Unknown", "version": -1}
    
        else:
            log("Missing pysssss.default.json, this extension may not work correctly. Please reinstall the extension.",
                type="ERROR", always=True, name="???")
            print(f"Extension path: {get_ext_dir()}")
            return {"name": "Unknown", "version": -1}

    with open(config_path, "r") as f:
        config = json.loads(f.read())
    return config

def link_js(src, dst):
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if os.name == "nt":
        try:
            import _winapi
            _winapi.CreateJunction(src, dst)
            return True
        except:
            pass
    try:
        os.symlink(src, dst)
        return True
    except:
        import logging
        logging.exception('')
        return False

def is_junction(path):
    if os.name != "nt":
        return False
    try:
        return bool(os.readlink(path))
    except OSError:
        return False

def install_js():
    src_dir = get_ext_dir("web/js")
    if not os.path.exists(src_dir):
        log("No JS")
        return

    should_install = should_install_js()
    if should_install:
        log("it looks like you're running an old version of ComfyUI that requires manual setup of web files, it is recommended you update your installation.", "warning", True)
    dst_dir = get_web_ext_dir()
    linked = os.path.islink(dst_dir) or is_junction(dst_dir)
    if linked or os.path.exists(dst_dir):
        if linked:
            if should_install:
                log("JS already linked")
            else:
                os.unlink(dst_dir)
                log("JS unlinked, PromptServer will serve extension")
        elif not should_install:
            shutil.rmtree(dst_dir)
            log("JS deleted, PromptServer will serve extension")
        return
    
    if not should_install:
        log("JS skipped, PromptServer will serve extension")
        return
    
    if link_js(src_dir, dst_dir):
        log("JS linked")
        return

    log("Copying JS files")
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

def should_install_js():
    return not hasattr(PromptServer.instance, "supports") or "custom_nodes_from_web" not in PromptServer.instance.supports


def init(check_imports=None):
    log("Init")

    if check_imports is not None:
        import importlib.util
        for imp in check_imports:
            spec = importlib.util.find_spec(imp)
            if spec is None:
                log(f"{imp} is required, please check requirements are installed.",
                    type="ERROR", always=True)
                return False

    install_js()
    return True


def get_async_loop():
    loop = None
    try:
        loop = asyncio.get_event_loop()
    except:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def get_http_session():
    loop = get_async_loop()
    return aiohttp.ClientSession(loop=loop)


async def download(url, stream, update_callback=None, session=None):
    close_session = False
    if session is None:
        close_session = True
        session = get_http_session()
    try:
        async with session.get(url) as response:
            size = int(response.headers.get('content-length', 0)) or None

            with tqdm(
                unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1], total=size,
            ) as progressbar:
                perc = 0
                async for chunk in response.content.iter_chunked(2048):
                    stream.write(chunk)
                    progressbar.update(len(chunk))
                    if update_callback is not None and progressbar.total is not None and progressbar.total != 0:
                        last = perc
                        perc = round(progressbar.n / progressbar.total, 2)
                        if perc != last:
                            last = perc
                            await update_callback(perc)
    finally:
        if close_session and session is not None:
            await session.close()


async def download_to_file(url, destination, update_callback=None, is_ext_subpath=True, session=None):
    if is_ext_subpath:
        destination = get_ext_dir(destination)
    with open(destination, mode='wb') as f:
        download(url, f, update_callback, session)


def wait_for_async(async_fn, loop=None):
    res = []

    async def run_async():
        r = await async_fn()
        res.append(r)

    if loop is None:
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

    loop.run_until_complete(run_async())

    return res[0]


def update_node_status(client_id, node, text, progress=None):
    if client_id is None:
        client_id = PromptServer.instance.client_id

    if client_id is None:
        return

    PromptServer.instance.send_sync("vlmnodes/update_status", {
        "node": node,
        "progress": progress,
        "text": text
    }, client_id)


async def update_node_status_async(client_id, node, text, progress=None):
    if client_id is None:
        client_id = PromptServer.instance.client_id

    if client_id is None:
        return

    await PromptServer.instance.send("vlmnodes/update_status", {
        "node": node,
        "progress": progress,
        "text": text
    }, client_id)


def get_config_value(key, default=None, throw=False):
    split = key.split(".")
    obj = get_extension_config()
    for s in split:
        if s in obj:
            obj = obj[s]
        else:
            if throw:
                raise KeyError("Configuration key missing: " + key)
            else:
                return default
    return obj


def is_inside_dir(root_dir, check_path):
    root_dir = os.path.abspath(root_dir)
    if not os.path.isabs(check_path):
        check_path = os.path.abspath(os.path.join(root_dir, check_path))
    return os.path.commonpath([check_path, root_dir]) == root_dir


def get_child_dir(root_dir, child_path, throw_if_outside=True):
    child_path = os.path.abspath(os.path.join(root_dir, child_path))
    if is_inside_dir(root_dir, child_path):
        return child_path
    if throw_if_outside:
        raise NotADirectoryError(
            "Saving outside the target folder is not allowed.")
    return None
