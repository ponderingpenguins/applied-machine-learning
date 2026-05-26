"""HTML rendering helpers for the model selection and model pages."""

from collections.abc import Iterable
from html import escape
from pathlib import Path
from string import Template

from gait_classification.utils import ModelType

TEMPLATES_DIR = Path(__file__).with_name("templates")


def _model_label(model_type: ModelType) -> str:
    return model_type.value.replace("_", " ").title()


def _render_template(template_name: str, **context: str) -> str:
    template_path = TEMPLATES_DIR / template_name
    template = Template(template_path.read_text(encoding="utf-8"))
    return template.safe_substitute(context)


def render_model_selection_page(
    model_types: Iterable[ModelType],
    selected_model: ModelType | None = None,
) -> str:
    model_options = list(model_types)
    selected_model = selected_model or model_options[0]
    selected_label = _model_label(selected_model)

    cards_html = "".join(
        f"""
        <article class=\"model-card {'selected' if model_type == selected_model else ''}\">
            <div class=\"model-card__eyebrow\">Available model</div>
            <h2>{escape(_model_label(model_type))}</h2>
            <p>Type: <strong>{escape(model_type.value)}</strong></p>
            <p>Use this model for embedding generation and classification.</p>
        </article>
        """
        for model_type in model_options
    )

    option_html = "".join(
        f'<option value="{escape(model_type.value)}" {"selected" if model_type == selected_model else ""}>{escape(_model_label(model_type))}</option>'
        for model_type in model_options
    )

    return _render_template(
        "model_selection.html",
        cards_html=cards_html,
        option_html=option_html,
        selected_label=escape(selected_label),
        selected_model_value=escape(selected_model.value),
    )


def render_model_page(selected_model: ModelType, status_message: str = "") -> str:
    return _render_template(
        "model_page.html",
        selected_label=escape(_model_label(selected_model)),
        selected_model_value=escape(selected_model.value),
        status_message=escape(status_message),
    )


def render_classify_user_page(selected_model: ModelType) -> str:
    return _render_template(
        "classify_user.html",
        selected_label=escape(_model_label(selected_model)),
        selected_model_value=escape(selected_model.value),
    )
