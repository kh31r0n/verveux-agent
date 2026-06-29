"""Node implementations for the restructured appointments graph.

Each node is a thin orchestrator: it reads from AgentState, calls the backend
through ``backend_client``, and emits Spanish replies. Business rules
(double-booking, version checks, audit) all live on the NestJS side.
"""
