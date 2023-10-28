"""The home page of the app."""

from calhacks2023 import styles
from calhacks2023.templates import template

import reflex as rx


@template(route="/", title="Home", image="/github.svg")
def index() -> rx.Component:
    """The home page.

    Returns:
        The UI for the home page.
    """
    with open("README.md") as readme:
        content = readme.read()
    return rx.markdown(content, component_map=styles.markdown_style)
