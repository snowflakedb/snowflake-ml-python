"""Utilities for generating consistent HTML representations across model classes."""

from typing import Any, Sequence

# Common CSS styles used across model classes
BASE_CONTAINER_STYLE = (
    "font-family: Helvetica, Arial, sans-serif; font-size: 14px; line-height: 1.5; "
    "color: #333; background-color: #f9f9f9; border: 1px solid #ddd; "
    "border-radius: 4px; padding: 15px; margin-bottom: 15px;"
)

HEADER_STYLE = "margin-top: 0; margin-bottom: 10px; font-size: 16px; color: #007bff;"

SECTION_HEADER_STYLE = "margin: 15px 0 10px; color: #007bff;"

GRID_CONTAINER_STYLE = "display: grid; grid-template-columns: 150px 1fr; gap: 10px;"

GRID_LABEL_STYLE = "font-weight: bold;"

DETAILS_STYLE = "margin: 5px 0; border: 1px solid #e0e0e0; border-radius: 4px;"

SUMMARY_STYLE = "padding: 8px; cursor: pointer; background-color: #f5f5f5; border-radius: 3px;"

SUMMARY_LABEL_STYLE = "font-weight: bold; color: #007bff;"

CONTENT_SECTION_STYLE = "margin-left: 10px;"

VERSION_ITEM_STYLE = "margin: 5px 0; padding: 5px; background-color: #f5f5f5; border-radius: 3px;"

NESTED_GROUP_STYLE = "border-left: 2px solid #e0e0e0; margin-left: 8px;"

ERROR_STYLE = "color: #888; font-style: italic;"


def create_base_container(title: str, content: str) -> str:
    """Create a base HTML container with consistent styling.

    Args:
        title: The main title for the container.
        content: The HTML content to include in the container.

    Returns:
        HTML string with the base container structure.
    """
    return f"""
    <div style="{BASE_CONTAINER_STYLE}">
        <h3 style="{HEADER_STYLE}">
            {title}
        </h3>
        {content}
    </div>
    """


def create_grid_section(items: list[tuple[str, str]]) -> str:
    """Create a grid layout section for key-value pairs.

    Args:
        items: List of (label, value) tuples to display in the grid.

    Returns:
        HTML string with grid layout.
    """
    grid_items = ""
    for label, value in items:
        grid_items += f"""
            <div style="{GRID_LABEL_STYLE}">{label}:</div>
            <div>{value}</div>
        """

    return f"""
    <div style="{GRID_CONTAINER_STYLE}">
        {grid_items}
    </div>
    """


def create_section_header(title: str) -> str:
    """Create a section header with consistent styling.

    Args:
        title: The section title.

    Returns:
        HTML string for the section header.
    """
    return f'<h4 style="{SECTION_HEADER_STYLE}">{title}</h4>'


def create_content_section(content: str) -> str:
    """Create a content section with consistent indentation.

    Args:
        content: The content to include in the section.

    Returns:
        HTML string with content section styling.
    """
    return f"""
    <div style="{CONTENT_SECTION_STYLE}">
        {content}
    </div>
    """


def create_collapsible_section(title: str, content: str, open_by_default: bool = True) -> str:
    """Create a collapsible section with consistent styling.

    Args:
        title: The title for the collapsible section.
        content: The content to include in the collapsible section.
        open_by_default: Whether the section should be open by default.

    Returns:
        HTML string for the collapsible section.
    """
    open_attr = "open" if open_by_default else ""

    return f"""
    <details style="{DETAILS_STYLE}" {open_attr}>
        <summary style="{SUMMARY_STYLE}">
            <span style="{SUMMARY_LABEL_STYLE}">{title}</span>
        </summary>
        <div style="padding: 10px; border-top: 1px solid #e0e0e0;">
            {content}
        </div>
    </details>
    """


def create_version_item(version_name: str, created_on: str, comment: str, is_default: bool = False) -> str:
    """Create a version item with consistent styling.

    Args:
        version_name: The name of the version.
        created_on: The creation timestamp.
        comment: The version comment.
        is_default: Whether this is the default version.

    Returns:
        HTML string for the version item.
    """
    default_text = " (Default)" if is_default else ""

    return f"""
    <div style="{VERSION_ITEM_STYLE}">
        <strong>Version:</strong> {version_name}{default_text}<br/>
        <strong>Created:</strong> {created_on}<br/>
        <strong>Comment:</strong> {comment}<br/>
    </div>
    """


def create_tag_item(tag_name: str, tag_value: str) -> str:
    """Create a tag item with consistent styling.

    Args:
        tag_name: The name of the tag.
        tag_value: The value of the tag.

    Returns:
        HTML string for the tag item.
    """
    return f"""
    <div style="margin: 5px 0;">
        <strong>{tag_name}:</strong> {tag_value}
    </div>
    """


def create_metric_item(metric_name: str, value: Any) -> str:
    """Create a metric item with consistent styling.

    Args:
        metric_name: The name of the metric.
        value: The value of the metric.

    Returns:
        HTML string for the metric item.
    """
    value_str = "N/A" if value is None else str(value)
    return f"""
    <div style="margin: 5px 0;">
        <strong>{metric_name}:</strong> {value_str}
    </div>
    """


def create_error_message(message: str) -> str:
    """Create an error message with consistent styling.

    Args:
        message: The error message to display.

    Returns:
        HTML string for the error message.
    """
    return f'<em style="{ERROR_STYLE}">{message}</em>'


def create_feature_spec_html(spec: Any, indent: int = 0) -> str:
    """Create HTML representation for a feature specification.

    Args:
        spec: The feature specification (FeatureSpec or FeatureGroupSpec).
        indent: The indentation level for nested features.

    Returns:
        HTML string for the feature specification.
    """
    # Import here to avoid circular imports
    from snowflake.ml.model._signatures import core

    if isinstance(spec, core.FeatureSpec):
        shape_str = f" shape={spec._shape}" if spec._shape else ""
        nullable_str = "" if spec._nullable else ", not nullable"
        return f"""
            <div style="margin: 3px 0; padding-left: {indent * 16}px;">
                <strong>{spec.name}</strong>: {spec._dtype}{shape_str}{nullable_str}
            </div>
        """
    elif isinstance(spec, core.FeatureGroupSpec):
        group_html = f"""
            <div style="margin: 8px 0;">
                <div style="margin: 3px 0; padding-left: {indent * 16}px;">
                    <strong>{spec.name}</strong> <span style="color: #666;">(group)</span>
                </div>
                <div style="border-left: 2px solid #e0e0e0; margin-left: {indent * 16 + 8}px;">
        """
        for sub_spec in spec._specs:
            group_html += create_feature_spec_html(sub_spec, indent + 1)
        group_html += """
                </div>
            </div>
        """
        return group_html
    return ""


def create_features_html(features: Sequence[Any], title: str) -> str:
    """Create HTML representation for a collection of features.

    Args:
        features: The sequence of feature specifications.
        title: The title for the feature collection.

    Returns:
        HTML string for the features collection.
    """
    if not features:
        return f"""
            <div style="margin: 5px 0; padding: 5px;">
                <em>No {title.lower()} features defined</em>
            </div>
        """

    html = """
        <div style="margin: 5px 0; padding: 5px;">
    """
    for feature in features:
        html += create_feature_spec_html(feature)
    html += "</div>"
    return html
