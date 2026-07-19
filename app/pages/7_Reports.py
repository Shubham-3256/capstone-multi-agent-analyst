"""Report compiling and download screen."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.components.cards import info_card
from app.components.sidebar import setup_page
from app.services.session import get_workflow_result, initialize_session


def resolve_path(p: str) -> Path | None:
    """Resolve workspace file paths robustly."""
    if not p:
        return None
    path_obj = Path(p)
    if path_obj.exists():
        return path_obj
    proj_rel = PROJECT_ROOT / p
    if proj_rel.exists():
        return proj_rel
    parts = Path(p).parts
    if parts and parts[0] == "workspace":
        trimmed = PROJECT_ROOT / "workspace" / "/".join(parts[1:])
        if trimmed.exists():
            return trimmed
        trimmed_direct = PROJECT_ROOT / "/".join(parts[1:])
        if trimmed_direct.exists():
            return trimmed_direct
    return None


def main() -> None:
    """Render the reports page."""
    initialize_session()
    setup_page("Generated Reports")

    st.title("📄 Export & Report Center")
    st.markdown("""
        Download the automatically compiled reports in standard enterprise formats. 
        All charts, data tables, and agent summaries are compiled, formatted, and embedded inside.
        """)
    st.divider()

    result = get_workflow_result()
    if not result:
        st.info(
            "No active workflow result. Please upload a dataset and run the pipeline on the Upload page."
        )
        return

    rep_result = getattr(result.state, "report_result", None)
    if not rep_result:
        st.warning("No generated report records found in the execution state.")
        return

    # 1. Report Metadata Cards
    st.subheader("Report Summary details")
    col1, col2, col3 = st.columns(3)
    with col1:
        info_card(
            "Document Title",
            rep_result.metadata.title,
            f"Author: {rep_result.metadata.author}",
            "📰",
        )
    with col2:
        info_card(
            "Template Type",
            rep_result.metadata.template_type.title(),
            "Layout configuration",
            "🎨",
        )
    with col3:
        info_card(
            "Compilation Time",
            f"{rep_result.duration_seconds:.2f}s",
            "HTML/PDF rendering duration",
            "⏱️",
        )

    st.divider()

    # 2. File Download Panel
    st.subheader("💾 Download Formats")

    col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)
    formats = [
        ("pdf", "📕 Export PDF Document", "application/pdf", col_dl1),
        (
            "docx",
            "📘 Export Word Docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            col_dl2,
        ),
        ("html", "🌐 Export HTML Page", "text/html", col_dl3),
        ("markdown", "📝 Export Markdown text", "text/markdown", col_dl4),
    ]

    for key, label, mime, col in formats:
        path_str = rep_result.output_paths.get(key, "")
        resolved = resolve_path(path_str)
        with col:
            if resolved and resolved.exists():
                try:
                    with open(resolved, "rb") as f:
                        file_bytes = f.read()
                    st.download_button(
                        label=label,
                        data=file_bytes,
                        file_name=resolved.name,
                        mime=mime,
                        key=f"dl_report_{key}",
                    )
                except Exception as e:
                    st.error(f"Failed to read {key} file: {e}")
            else:
                st.warning(f"{key.upper()} path not resolved or file missing.")

    st.divider()

    # 3. Report Preview
    md_path_str = rep_result.output_paths.get("markdown", "")
    md_path = resolve_path(md_path_str)
    if md_path and md_path.exists():
        st.subheader("📝 Report Content Preview")
        try:
            with open(md_path, encoding="utf-8") as f:
                md_content = f.read()
            with st.container(border=True):
                st.markdown(md_content)
        except Exception as e:
            st.error(f"Could not load markdown preview: {e}")

    st.divider()

    # 4. Report Manifest / Provenance Checks
    st.subheader("🛡️ Cryptographic Report Manifest & Audit Checks")
    manifest = rep_result.manifest
    with st.expander("Show manifest & verification hashes", expanded=False):
        st.markdown(f"""
            * **Report UUID Token:** `{manifest.report_id}`
            * **Dataset SHA256 Signature:** `{manifest.dataset_hash}`
            * **Workflow Version:** `{manifest.pipeline_version}`
            * **Underlying Core LLM Engine:** `{manifest.model_version}`
            * **Included Chapters/Sections:** {manifest.sections}
            * **Report Generation Timestamp (UTC):** `{manifest.generation_timestamp}`
            * **Embedded Plot Assets:** {manifest.charts_included}
            """)


if __name__ == "__main__":
    main()
