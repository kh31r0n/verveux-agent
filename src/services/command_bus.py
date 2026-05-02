
import httpx
import uuid
from typing import Any, Dict

from src.config import settings

# Assuming CommandResult is defined in a shared schema, for now we define it here
class CommandResult:
    def __init__(self, success: bool, data: Any = None, error: Dict[str, Any] = None):
        self.success = success
        self.data = data
        self.error = error

class CommandBusClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key

    async def send_command(self, command: str, tenant_id: str, contact_id: str, conversation_id: str, payload: Dict[str, Any]) -> CommandResult:
        idempotency_key = str(uuid.uuid4())
        headers = {
            'x-agent-key': self.api_key,
            'Content-Type': 'application/json',
        }
        # TODO: Implement camelCase conversion for the payload
        json_payload = {
            'command': command,
            'tenantId': tenant_id,
            'contactId': contact_id,
            'conversationId': conversation_id,
            'payload': payload,
            'idempotencyKey': idempotency_key,
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(f'{self.base_url}/api/v1/internal/commands', json=json_payload, headers=headers)
                response.raise_for_status()  # Raise an exception for non-2xx status codes
                return CommandResult(success=True, data=response.json())
        except httpx.HTTPStatusError as e:
            return CommandResult(success=False, error={'code': f'HTTP_{e.response.status_code}', 'message': str(e)})
        except httpx.RequestError as e:
            return CommandResult(success=False, error={'code': 'NETWORK_ERROR', 'message': str(e)})


command_bus_client = CommandBusClient(base_url=settings.nestjs_base_url, api_key=settings.webhook_api_key)
