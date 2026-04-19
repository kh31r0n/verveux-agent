import asyncio
import json
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import psycopg

async def main():
    conn_string = "postgresql://postgres:postgres@localhost:5432/helena?sslmode=disable"
    async with await psycopg.AsyncConnection.connect(conn_string) as conn:
        checkpointer = AsyncPostgresSaver(conn)
        # get any checkpoint
        async for c in checkpointer.alist(None):
            checkpoint = c.checkpoint
            state = checkpoint.get("channel_values", {})
            print(f"thread: {checkpoint.get('thread_id')}")
            print(f"contact_id: {state.get('contact_id')}")
            break

asyncio.run(main())
