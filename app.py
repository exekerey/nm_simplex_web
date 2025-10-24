import csv
import io
import json
from typing import List, Tuple

import streamlit as st

from simplex_solver import SimplexError, SimplexResult, simplex


st.set_page_config(page_title="Simplex Solver", layout="wide")
st.title("Simplex Method Solver")
st.markdown(
    """
Solve maximization problems of the form
\\( \\max c^T x \\) subject to \\( A x \\le b \\) and \\( x \\ge 0 \\).
Enter data manually or upload a JSON/CSV description.
"""
)


def _display_tableau(result: SimplexResult) -> None:
    n = result.num_variables
    m = result.num_constraints
    column_labels = (
        [f"x{j + 1}" for j in range(n)]
        + [f"s{j + 1}" for j in range(m)]
        + ["rhs"]
    )

    rows = []
    for idx, row in enumerate(result.tableau[:-1]):
        entry = {"row": f"C{idx + 1}"}
        for col_idx, label in enumerate(column_labels):
            entry[label] = round(row[col_idx], 6)
        rows.append(entry)

    objective_row = {"row": "Obj"}
    for col_idx, label in enumerate(column_labels):
        objective_row[label] = round(result.tableau[-1][col_idx], 6)
    rows.append(objective_row)

    st.table(rows)


def _display_result(result: SimplexResult) -> None:
    if result.status == "optimal":
        st.success(
            f"Optimal solution found in {result.iterations} iterations. "
            f"Objective value: {round(result.optimal_value, 6)}"
        )
        st.write("Decision variables:", [round(x, 6) for x in result.solution or []])
    elif result.status == "unbounded":
        st.error("The LP is unbounded.")
    elif result.status == "iteration_limit":
        st.error("Iteration limit reached; no optimal solution identified.")
    else:
        st.error(f"Solver status: {result.status}")

    st.caption(f"Current basis: {result.basis}")
    st.markdown("#### Final Tableau")
    _display_tableau(result)


def _run_solver(c: List[float], A: List[List[float]], b: List[float]) -> None:
    try:
        result = simplex(c, A, b)
    except SimplexError as exc:
        st.error(f"Input error: {exc}")
        return
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Unexpected error: {exc}")
        return

    _display_result(result)


def _manual_tab():
    st.subheader("Manual Input")
    with st.form("manual_form"):
        n_vars = st.number_input("Number of decision variables", min_value=1, value=3, step=1)
        n_cons = st.number_input("Number of constraints", min_value=1, value=3, step=1)

        st.markdown("##### Objective Coefficients (c)")
        c_values: List[float] = []
        for j in range(int(n_vars)):
            value = st.number_input(
                f"c{j + 1}",
                value=0.0,
                step=1.0,
                key=f"c_{j}",
            )
            c_values.append(float(value))

        st.markdown("##### Constraint Matrix (A) and RHS (b)")
        A_values: List[List[float]] = []
        b_values: List[float] = []
        for i in range(int(n_cons)):
            cols = st.columns(int(n_vars) + 1)
            row: List[float] = []
            for j in range(int(n_vars)):
                value = cols[j].number_input(
                    f"a[{i + 1},{j + 1}]",
                    value=0.0,
                    step=1.0,
                    key=f"a_{i}_{j}",
                )
                row.append(float(value))
            b_val = cols[-1].number_input(
                f"b{i + 1}",
                value=0.0,
                step=1.0,
                key=f"b_{i}",
            )
            A_values.append(row)
            b_values.append(float(b_val))

        submitted = st.form_submit_button("Solve")

    if submitted:
        _run_solver(c_values, A_values, b_values)


def _parse_json_payload(payload: str) -> Tuple[List[float], List[List[float]], List[float]]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SimplexError(f"Invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise SimplexError("JSON root must be an object containing c, A, b.")
    missing = [key for key in ("c", "A", "b") if key not in data]
    if missing:
        raise SimplexError(f"JSON missing keys: {', '.join(missing)}.")

    return data["c"], data["A"], data["b"]


def _parse_csv_payload(payload: str) -> Tuple[List[float], List[List[float]], List[float]]:
    reader = csv.DictReader(io.StringIO(payload))
    if reader.fieldnames is None:
        raise SimplexError("CSV must include a header row.")

    fieldnames_lower = [field.lower() for field in reader.fieldnames]
    if "type" not in fieldnames_lower or "b" not in fieldnames_lower:
        raise SimplexError("CSV must include columns named 'type' and 'b'.")

    type_field = reader.fieldnames[fieldnames_lower.index("type")]
    b_field = reader.fieldnames[fieldnames_lower.index("b")]
    coefficient_headers = [
        field
        for idx, field in enumerate(reader.fieldnames)
        if idx not in (fieldnames_lower.index("type"), fieldnames_lower.index("b"))
    ]
    if not coefficient_headers:
        raise SimplexError("CSV must include at least one decision variable column.")

    objective_rows = []
    constraints = []

    for raw_row in reader:
        row_type = (raw_row.get(type_field) or "").strip().lower()
        if row_type == "objective":
            objective_rows.append(raw_row)
        elif row_type == "constraint":
            constraints.append(raw_row)
        else:
            raise SimplexError("CSV 'type' column must be 'objective' or 'constraint'.")

    if len(objective_rows) != 1:
        raise SimplexError("CSV must contain exactly one objective row.")
    if not constraints:
        raise SimplexError("CSV must include at least one constraint row.")

    def _coefficients(row):
        coeffs = []
        for header in coefficient_headers:
            value = row.get(header, "").strip()
            coeffs.append(float(value) if value else 0.0)
        return coeffs

    c = _coefficients(objective_rows[0])
    A = [_coefficients(row) for row in constraints]
    b = []
    for row in constraints:
        value = (row.get(b_field) or "").strip()
        if value == "":
            raise SimplexError("Constraint rows must include a value for b.")
        b.append(float(value))

    return c, A, b


def _file_upload_tab():
    st.subheader("Upload JSON or CSV")
    if "json_editor" not in st.session_state:
        st.session_state["json_editor"] = json.dumps(
            {
                "c": [3, 5],
                "A": [[2, 1], [1, 3]],
                "b": [14, 10],
            },
        )
    if "csv_problem" not in st.session_state:
        st.session_state["csv_problem"] = None
    if "csv_filename" not in st.session_state:
        st.session_state["csv_filename"] = ""

    uploaded = st.file_uploader(
        "Upload file", type=["json", "csv"], accept_multiple_files=False
    )

    if uploaded:
        payload = uploaded.getvalue().decode("utf-8")
        if uploaded.name.lower().endswith(".json"):
            st.session_state["json_editor"] = payload
            st.session_state["csv_problem"] = None
            st.session_state["csv_filename"] = ""
            st.success(f"Loaded JSON file '{uploaded.name}' into the editor.")
        else:
            try:
                parsed_csv = _parse_csv_payload(payload)
            except SimplexError as exc:
                st.session_state["csv_problem"] = None
                st.session_state["csv_filename"] = ""
                st.error(f"File error: {exc}")
            else:
                st.session_state["csv_problem"] = parsed_csv
                st.session_state["csv_filename"] = uploaded.name
                st.success(f"Parsed CSV file '{uploaded.name}'.")
    else:
        # Clear stored CSV data when no file is selected.
        st.session_state["csv_problem"] = None
        st.session_state["csv_filename"] = ""

    st.caption(
        "JSON example: `{ \"c\": [3, 5], \"A\": [[1, 0], [0, 2]], \"b\": [4, 12] }`. "
        "CSV header example: `type,x1,x2,b`."
    )

    json_content = st.text_area(
        "JSON Problem Definition",
        key="json_editor",
        placeholder="Paste or edit JSON for the LP here...",
        height=260,
    )

    json_status = st.empty()
    parsed_json_problem = None
    if json_content.strip():
        try:
            parsed_json_problem = _parse_json_payload(json_content)
        except SimplexError as exc:
            json_status.error(f"JSON error: {exc}")
        else:
            json_status.success("JSON input is valid.")
    else:
        json_status.info("Provide JSON above or use the file uploader.")

    if st.session_state["csv_filename"]:
        st.caption(f"Active CSV file: {st.session_state['csv_filename']}")

    col_json, col_csv = st.columns(2)
    with col_json:
        if st.button("Solve JSON Problem", use_container_width=True):
            if parsed_json_problem:
                _run_solver(*parsed_json_problem)
            else:
                st.error("Provide a valid JSON problem before solving.")
    with col_csv:
        if st.button("Solve CSV Problem", use_container_width=True):
            csv_problem = st.session_state.get("csv_problem")
            if csv_problem:
                _run_solver(*csv_problem)
            else:
                st.error("Upload and parse a CSV file first.")


tab_manual, tab_file = st.tabs(["Manual Input", "File Upload"])
with tab_manual:
    _manual_tab()
with tab_file:
    _file_upload_tab()
