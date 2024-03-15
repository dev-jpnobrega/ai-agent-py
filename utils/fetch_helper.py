import json
from typing import Any, Dict, Optional

import aiohttp
import httpx as http
import requests


class IConfigureTimeoutResponse:
    def __init__(self, signal, timeout_id):
        self.signal = signal
        self.timeout_id = timeout_id


# def configure_timeout(timeout: int) -> IConfigureTimeoutResponse:
#     timeout_id = None
#     signal = None
#     if timeout:
#         signal = None  # To be replaced with actual AbortController in Python
#         # Convert milliseconds to seconds
#         timeout_id = signal.timeout(timeout / 1000)
#     return IConfigureTimeoutResponse(signal, timeout_id)


# def try_parse_json(value: Any) -> Any:
#     try:
#         return json.loads(value)
#     except ValueError:
#         return value


# class IResponseHeaders(Dict[str, str]):
#     pass


# class IResponse:
#     body = str
#     response = object


# async def format_response(response) -> IResponse:
#     body = try_parse_json(await response.text())
#     response_headers = IResponseHeaders(response.headers)

#     formatted_response = {
#         "body": body,
#         "response": {
#             "body": body,
#             "headers": response_headers,
#             "ok": response.ok,
#             "statusCode": response.status,
#             "statusText": response.reason,
#         },
#     }

#     return formatted_response if response.ok else {"body": "Request Error"}


# async def fetch_openapi(data: Dict[str, Any], timeout: int, headers: Dict[str, str] = None) -> IResponse:
#     timeout_response = configure_timeout(timeout)

#     async with aiohttp.ClientSession() as session:
#         async with session.post(data.get("url"), method=data.get("requestMethod"), headers=headers, data=json.dumps(data.get("data")), signal=timeout_response.signal) as response:
#             clearTimeout(timeout_response.timeout_id)
#             return await format_response(response)


def fetch(url: str, method: str, data: Dict[str, Any] = None, headers: Dict[str, str] = None) -> Optional[Dict[str, Any]]:
    """
    Faz uma solicitação HTTP para a URL fornecida.

    Args:
        url (str): A URL para fazer a solicitação.
        method (str): O método HTTP a ser usado (por exemplo, 'GET', 'POST').
        data (Dict[str, Any], opcional): O corpo da solicitação em formato JSON. Padrão é None.
        headers (Dict[str, str], opcional): Os cabeçalhos da solicitação. Padrão é None.

    Returns:
        Dict[str, Any] or None: Os dados da resposta JSON, se a solicitação for bem-sucedida. None caso contrário.
    """
    try:
        if method.upper() == 'GET':
            response = requests.get(url, headers=headers)
        elif method.upper() == 'POST':
            response = requests.post(url, json=data, headers=headers)
        else:
            raise ValueError("Método HTTP inválido. Use 'GET' ou 'POST'.")

        response.raise_for_status()  # Lança uma exceção para erros HTTP
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Erro ao fazer a solicitação para {url}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Erro ao decodificar JSON da resposta de {url}: {e}")
        return None
    except ValueError as e:
        print(f"Erro: {e}")
        return None
