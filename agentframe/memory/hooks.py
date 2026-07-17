"""
agentframe persistence hooks.

agentframe does not implement persistence itself. Instead, it directly accepts
LangGraph's BaseCheckpointSaver, which is passed through to the compiled graph.

Usage:

    from langgraph.checkpoint.sqlite import SqliteSaver
    from agentframe import Agent

    checkpointer = SqliteSaver.from_conn_string("sessions.db")
    agent = Agent(model="gpt-4o", checkpointer=checkpointer)

    # Session management is handled via config["configurable"]["thread_id"]
    agent.invoke("hello", config={"configurable": {"thread_id": "session-1"}})
    agent.invoke("follow up", config={"configurable": {"thread_id": "session-1"}})

Available checkpointers from langgraph:

    - langgraph.checkpoint.memory.MemorySaver
    - langgraph.checkpoint.sqlite.SqliteSaver
    - langgraph.checkpoint.postgres.PostgresSaver
    - langgraph.checkpoint.aiosqlite.AioSqliteSaver

You can also implement your own by subclassing BaseCheckpointSaver.
"""
