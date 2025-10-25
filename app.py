import csv
import io
import json
from typing import List, Tuple

import streamlit as st

from simplex_solver import SimplexError, SimplexResult, simplex


st.set_page_config(page_title="simplex solver", layout="wide")
st.title("simplex method solver (standard form only)")

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
        entry = {"row": f"c{idx + 1}"}
        for col_idx, label in enumerate(column_labels):
            entry[label] = round(row[col_idx], 6)
        rows.append(entry)

    objective_row = {"row": "obj"}
    for col_idx, label in enumerate(column_labels):
        objective_row[label] = round(result.tableau[-1][col_idx], 6)
    rows.append(objective_row)

    st.table(rows)


def _display_result(result: SimplexResult) -> None:
    if result.status == "optimal":
        st.success(
            f"optimal solution found in {result.iterations} iterations. "
            f"objective value: {round(result.optimal_value, 6)}"
        )
        st.write("decision variables:", [round(x, 6) for x in result.solution or []])
    elif result.status == "unbounded":
        st.error("the lp is unbounded.")
    elif result.status == "iteration_limit":
        st.error("iteration limit reached; no optimal solution identified.")
    else:
        st.error(f"solver status: {result.status}")

    st.caption(f"current basis: {result.basis}")
    st.markdown("#### final tableau")
    _display_tableau(result)


def _run_solver(c: List[float], A: List[List[float]], b: List[float]) -> None:
    try:
        result = simplex(c, A, b)
    except SimplexError as exc:
        st.error(f"input error: {exc}")
        return
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"unexpected error: {exc}")
        return

    _display_result(result)


def _manual_tab():
    st.subheader("manual input")
    with st.form("manual_form"):
        n_vars = st.number_input("number of decision variables", min_value=1, value=3, step=1)
        n_cons = st.number_input("number of constraints", min_value=1, value=3, step=1)

        st.markdown("##### objective function coefficients")
        c_values: List[float] = []
        for j in range(int(n_vars)):
            value = st.number_input(
                f"c{j + 1}",
                value=0.0,
                step=1.0,
                key=f"c_{j}",
            )
            c_values.append(float(value))

        st.markdown("##### constraint matrix (A) and RHS (b)")
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

        submitted = st.form_submit_button("solve")

    if submitted:
        _run_solver(c_values, A_values, b_values)


def _parse_json_payload(payload: str) -> Tuple[List[float], List[List[float]], List[float]]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SimplexError(f"invalid json: {exc}") from exc

    if not isinstance(data, dict):
        raise SimplexError("json root must be an object containing c, A, b.")
    missing = [key for key in ("c", "A", "b") if key not in data]
    if missing:
        raise SimplexError(f"json missing keys: {', '.join(missing)}.")

    return data["c"], data["A"], data["b"]


def _parse_csv_payload(payload: str) -> Tuple[List[float], List[List[float]], List[float]]:
    reader = csv.DictReader(io.StringIO(payload))
    if reader.fieldnames is None:
        raise SimplexError("csv must include a header row.")

    fieldnames_lower = [field.lower() for field in reader.fieldnames]
    if "type" not in fieldnames_lower or "b" not in fieldnames_lower:
        raise SimplexError("csv must include columns named 'type' and 'b'.")

    type_field = reader.fieldnames[fieldnames_lower.index("type")]
    b_field = reader.fieldnames[fieldnames_lower.index("b")]
    coefficient_headers = [
        field
        for idx, field in enumerate(reader.fieldnames)
        if idx not in (fieldnames_lower.index("type"), fieldnames_lower.index("b"))
    ]
    if not coefficient_headers:
        raise SimplexError("csv must include at least one decision variable column.")

    objective_rows = []
    constraints = []

    for raw_row in reader:
        row_type = (raw_row.get(type_field) or "").strip().lower()
        if row_type == "objective":
            objective_rows.append(raw_row)
        elif row_type == "constraint":
            constraints.append(raw_row)
        else:
            raise SimplexError("csv 'type' column must be 'objective' or 'constraint'.")

    if len(objective_rows) != 1:
        raise SimplexError("csv must contain exactly one objective row.")
    if not constraints:
        raise SimplexError("csv must include at least one constraint row.")

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
            raise SimplexError("constraint rows must include a value for b.")
        b.append(float(value))

    return c, A, b


def _file_upload_tab():
    st.subheader("upload json or csv")
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
        "upload file", type=["json", "csv"], accept_multiple_files=False
    )

    if uploaded:
        payload = uploaded.getvalue().decode("utf-8")
        if uploaded.name.lower().endswith(".json"):
            st.session_state["json_editor"] = payload
            st.session_state["csv_problem"] = None
            st.session_state["csv_filename"] = ""
            st.success(f"loaded json file '{uploaded.name}' into the editor.")
        else:
            try:
                parsed_csv = _parse_csv_payload(payload)
            except SimplexError as exc:
                st.session_state["csv_problem"] = None
                st.session_state["csv_filename"] = ""
                st.error(f"file error: {exc}")
            else:
                st.session_state["csv_problem"] = parsed_csv
                st.session_state["csv_filename"] = uploaded.name
                st.success(f"parsed csv file '{uploaded.name}'.")
    else:
        st.session_state["csv_problem"] = None
        st.session_state["csv_filename"] = ""

    st.caption(
        "json example: `{ \"c\": [3, 5], \"A\": [[1, 0], [0, 2]], \"b\": [4, 12] }`. "
        "csv header example: `type,x1,x2,b`."
    )

    json_content = st.text_area(
        "json problem definition",
        key="json_editor",
        placeholder="paste or edit json for the lp here...",
        height=260,
    )

    json_status = st.empty()
    parsed_json_problem = None
    if json_content.strip():
        try:
            parsed_json_problem = _parse_json_payload(json_content)
        except SimplexError as exc:
            json_status.error(f"json error: {exc}")
        else:
            json_status.success("json input is valid.")
    else:
        json_status.info("provide json above or use the file uploader.")

    if st.session_state["csv_filename"]:
        st.caption(f"active csv file: {st.session_state['csv_filename']}")

    if st.button("solve problem", use_container_width=True):
        csv_problem = st.session_state.get("csv_problem")
        if csv_problem:
            _run_solver(*csv_problem)
        elif parsed_json_problem:
            _run_solver(*parsed_json_problem)
        else:
            st.error("provide a valid json problem or upload a csv file before solving.")


tab_manual, tab_file = st.tabs(["manual input", "file upload"])
with tab_manual:
    _manual_tab()
with tab_file:
    _file_upload_tab()
