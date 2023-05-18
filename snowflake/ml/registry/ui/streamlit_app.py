import os
import sys
from logging import getLogger
from typing import Optional, cast

import streamlit as st

from snowflake import connector
from snowflake.snowpark import session

_logger = getLogger(__name__)

# TODO(sdas): Temporary hack till we make
# [streamlit work with bazel](https://github.com/streamlit/streamlit/issues/4576)
filename = os.path.abspath(__file__)
pos = filename.find("/snowflake/ml/")
REPO_ROOT = filename[0:pos]
if REPO_ROOT not in sys.path:
    _logger.info("Adding %s to system path" % REPO_ROOT)
    sys.path.append(REPO_ROOT)

from snowflake.ml.registry import model_registry  # NOQA
from snowflake.ml.registry.ui import render_pandas_df  # NOQA
from snowflake.ml.utils import connection_params  # NOQA

# Set page level parameters. This needs to be the first streamlit command run.
st.set_page_config(
    page_title="Snowflake Model Registry Browser",
    page_icon="https://www.snowflake.com/etc.clientlibs/snowflake-site/clientlibs/clientlib-react/resources/"
    + "favicon-32x32.png?v=2",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://developers.snowflake.com",
        "About": "This is a simple UI layer on top of Snowflake's Model Registry powered by Snowpark Streamlit",
    },
)


@st.cache_resource  # type: ignore
def create_session() -> session.Session:
    sf_session = session.Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()
    return cast(session.Session, sf_session)


def get_registry(session: session.Session, name: Optional[str] = None) -> model_registry.ModelRegistry:
    if name:
        registry = model_registry.ModelRegistry(session=session, database_name=name)
    else:
        registry = model_registry.ModelRegistry(session=session)
    return registry


def clear_model_id() -> None:
    st.session_state.model_id_placeholder.empty()
    if "model_id" in st.session_state:
        del st.session_state.model_id


def clear_model() -> None:
    clear_model_id()
    st.session_state.model_name_placeholder.empty()
    if "model_name" in st.session_state:
        del st.session_state.model_name


def clear_registry() -> None:
    clear_model_id()
    clear_model()
    if "registry" in st.session_state:
        del st.session_state.registry
    if "registry_name" in st.session_state and st.session_state.registry_name:
        st.experimental_set_query_params(registry=st.session_state.registry_name)
    else:
        st.experimental_set_query_params()


def reopen_registry() -> None:
    _logger.info("Called reopen_registry()")
    if "registry_name" in st.session_state:
        registry_name = st.session_state.registry_name
    else:
        registry_name = None
    try:
        registry = get_registry(session=st.session_state.sf_session, name=registry_name)
        st.session_state.registry = registry
        params = st.experimental_get_query_params()
        if registry_name:
            params["registry"] = registry_name
        elif "registry" in params:
            del params["registry"]
        st.experimental_set_query_params(**params)
    except connector.DataError:
        clear_registry()


def display_main_view() -> None:
    if "registry" in st.session_state:
        params = {}
        if "registry_name" in st.session_state and st.session_state.registry_name:
            params["registry"] = st.session_state.registry_name
        if "model_id" in st.session_state:
            params["id"] = st.session_state.model_id
        if "model_name" in st.session_state:
            params["model"] = st.session_state.model_name
        renderer = render_pandas_df.DfRenderer(registry=st.session_state.registry, url_params=params)
        content = renderer.render()
        st.session_state.header_placeholder.header(content.header)
        if content.found:
            with st.session_state.content_placeholder.container():
                st.write(content.description, unsafe_allow_html=True)
                if content.history:
                    st.header(content.history_header)
                    st.write(content.history, unsafe_allow_html=True)
        else:
            st.session_state.content_placeholder.error("No such model found.")
    else:
        if "registry_name" in st.session_state and st.session_state.registry_name:
            st.session_state.content_placeholder.error("Registry (%s) does not exist." % st.session_state.registry_name)
        else:
            st.session_state.content_placeholder.error("Default Registry does not exist.")


def init_server() -> None:
    # Set page level parameters
    st.set_page_config(
        page_title="Snowflake Model Registry Browser",
        page_icon="https://www.snowflake.com/etc.clientlibs/snowflake-site/clientlibs/clientlib-react/resources/"
        + "favicon-32x32.png?v=2",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://developers.snowflake.com",
            "About": "This is a simple UI layer on top of Snowflake's Model Registry powered by Snowpark Streamlit",
        },
    )
    st.title("SnowML Model Registry Browser")
    params = st.experimental_get_query_params()
    registry_name = params["registry"][0] if "registry" in params else ""
    with st.sidebar:
        st.text_input(
            "Registry",
            key="registry_name",
            value=registry_name,
            placeholder="Default",
            help="Enter registry name if not default.",
            on_change=reopen_registry,
        )
        st.session_state.model_name_placeholder = st.empty()
        if "model" in params and params["model"]:
            model_name = params["model"][0]
            st.session_state.model_name_placeholder.text_input(
                "Model Name",
                key="model_name",
                value=model_name,
                help="Enter a valid model name.",
                disabled=True,
            )

        st.session_state.model_id_placeholder = st.empty()
        if "id" in params and params["id"]:
            model_id = params["id"][0]
            st.session_state.model_id_placeholder.text_input(
                "Model Id",
                key="model_id",
                value=model_id,
                help="Enter a valid model ID.",
                disabled=True,
            )

    st.session_state.header_placeholder = st.empty()
    st.session_state.content_placeholder = st.empty()


def display_content() -> None:
    with st.spinner("Loading ..."):
        if "sf_session" not in st.session_state:
            st.session_state.sf_session = create_session()
        reopen_registry()
        display_main_view()


if __name__ == "__main__":
    init_server()
    display_content()
