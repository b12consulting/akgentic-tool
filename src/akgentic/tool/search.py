from typing import Any, Callable, Literal

from pydantic import BaseModel
from tavily import TavilyClient

from akgentic.tool.core import BaseTool


class WebSearch(BaseModel):
    max_results: int = 5


class SearchTool(BaseTool):
    def get_tools(self) -> list[Callable]:

        tools = []

        for params in self.tool_card.params or []:
            if isinstance(params, WebSearch):
                web_search: Callable = self.web_search_factory(params)
                tools.append(web_search)

        return tools

    def web_search_factory(self, params: WebSearch) -> Callable:

        def web_search_tool(
            query: str,
            search_depth: Literal["basic", "advanced"] = "basic",
        ) -> Any:
            """Search the internet for general web results based on a query.
            Use this function to search for information that is not available in the vector store.
            The `web_search_tool` is particularly useful for answering
            questions about current events.
            """

            tavily_client = TavilyClient()
            return tavily_client.search(
                query,
                max_results=params.max_results,
                search_depth=search_depth,
            )

        return web_search_tool
