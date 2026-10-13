"""Shared product/evaluation assembly and ownership of its request resources."""


class _IndexConnections:
    def __init__(self, index):
        self.index = index

    def close(self):
        self.index.db.close()
        if self.index._writer is not None:
            self.index._writer.close()


def _close_agent(agent):
    """Close only resources created by assembly; injected dependencies remain caller-owned."""
    resources = getattr(agent, "_owned_resources", [])
    agent._owned_resources = []
    seen = set()
    for resource in reversed(resources):
        if resource is None or id(resource) in seen:
            continue
        seen.add(id(resource))
        close = getattr(resource, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass  # finish the remaining cleanup without replacing an evaluation failure


def _assemble(project="default", on_event=None, index=None, *, web=True, index_home=None,
              roots=None, web_client=None, llm=None, searcher=None, record=True):
    from peresearch.agent import Agent, Limits
    from peresearch.llm import LLM
    from peresearch.papers import Papers
    from peresearch.tools import Toolbox
    from peresearch.web import Web
    from peresearch.wiki import Wikipedia
    from peresearch.zetokrag.index import Index
    from peresearch.zetokrag.search import Searcher

    on_event = on_event or (lambda kind, detail: None)
    owned = []
    holder = type("Resources", (), {})()
    holder._owned_resources = owned
    try:
        if llm is None:
            llm = LLM(on_wait=lambda message: on_event("wait", message))
            owned.append(getattr(llm, "client", None))
        if searcher is None:
            if index is None:
                index = Index() if index_home is None else Index(index_home)
                owned.append(_IndexConnections(index))
                owned.append(getattr(getattr(index, "_embedder", None), "client", None))
            searcher = Searcher(index)
            owned.append(getattr(getattr(searcher, "reranker", None), "client", None))
        pages = web_client
        if web and pages is None:
            pages = Web()
            owned.extend([getattr(pages, "http", None), pages])
        papers = Papers() if web else None
        if papers is not None:
            owned.extend(source.http for source in papers.sources)
        wiki = Wikipedia() if web else None
        if wiki is not None:
            owned.append(wiki.http)
        toolbox = Toolbox(searcher, pages if web else None, roots=roots, papers=papers, wiki=wiki)
        toolbox.local_only = not web
        agent = Agent(llm, toolbox, project=project, on_event=on_event,
                      limits=Limits.from_settings(), record=record)
        agent._owned_resources = owned
        return agent
    except BaseException:
        _close_agent(holder)
        raise
