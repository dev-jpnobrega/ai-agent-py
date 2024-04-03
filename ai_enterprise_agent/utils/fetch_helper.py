import json
from typing import Any, Dict, Optional

import requests


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

        response.raise_for_status()
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
