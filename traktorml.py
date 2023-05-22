from collections import defaultdict
from typing import List, Tuple, Union, Dict

import mlflow
from pandas import DataFrame
from pydantic import BaseModel
from textual.containers import ScrollableContainer
from textual.events import Key
from textual.widgets import DataTable, Footer
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Header, Select
from dotenv import load_dotenv

load_dotenv()

MISSING_VALUE = "N/A"


class Experiment(BaseModel):
    """An MLFlow experiment."""
    name: str
    id: str


class RunInfo(BaseModel):
    """An MLFlow run."""
    metrics: dict
    params: dict
    tags: dict

def get_experiments() -> list[Experiment]:
    return [
        Experiment(name=experiment.name, id=experiment.experiment_id)
        for experiment in mlflow.search_experiments()
    ]


def get_experiment_runs_df(experiment: Experiment) -> DataFrame:
    return  mlflow.search_runs(experiment_names=[experiment.name]).fillna(MISSING_VALUE)


class ExperimentRunsView(BaseModel):
    """A view of the runs of an MLFlow experiment."""
    experiment_runs_df: DataFrame
    deselected_columns: List[str] = []
    deselected_run_ids: List[str] = []
    current_sort: Union[Tuple[str, bool], None] = None
    filter_selected_values: dict = {}
    filter_deselected_values: Dict[str, list] = defaultdict(list)

    class Config:
        arbitrary_types_allowed = True

    @property
    def selected_columns(self) -> List[str]
        return [
            c for c in self.experiment_runs_df.columns
            if c not in self.deselected_columns
        ]

    @property
    def selected_runs_df(self) -> DataFrame:
        runs_df = self.experiment_runs_df[self.selected_columns]
        if self.deselected_run_ids:
            runs_df = runs_df[
                ~self.experiment_runs_df["run_id"].isin(self.deselected_run_ids)
            ]
        for col, val in self.filter_selected_values.items():
            runs_df = runs_df[runs_df[col] == val]
        for col, vals in self.filter_deselected_values.items():
            runs_df = runs_df[~runs_df[col].isin(vals)]
        return runs_df

    @property
    def selected_data(self) -> List[Tuple]:
        return [tuple(row) for row in self.selected_runs_df.values]

    def reset_view(self) -> None:
        self.deselected_columns = []
        self.deselected_run_ids = []
        self.current_sort = None
        self.filter_selected_values = {}
        self.filter_deselected_values = defaultdict(list)


def set_data_table_to_experiment_view(experiment_view: ExperimentRunsView, data_table: DataTable) -> None:
    """Helper function to set the values of a Textual data table to an experiment view."""
    data_table.clear(columns=True)
    for col in experiment_view.selected_columns:
        data_table.add_column(label=col, key=col)
    for i, row in enumerate(experiment_view.selected_data):
        data_table.add_row(*row, key=row[0])


class TraktorML(App):

    TITLE = "Traktor ML"
    CSS = "DataTable {height: 1fr}"
    BINDINGS = [
        ("s", "sort", "Sort by column"),
        ("p", "clear_selections", "Clear selections"),
        ("f", "filter_on_values", "Filter on values"),
        ("d", "deselect_values", "Deselect values"),
        ("c", "remove_columns", "Remove columns"),
        ("r", "remove_row", "Remove Row"),
        ("q", "quit", "Quit"),
    ]

    experiments = get_experiments()
    selected_experiment: Union[Experiment, None] = None
    experiment_views: Dict[str, ExperimentRunsView] = {}

    @property
    def experiment_view(self) -> ExperimentRunsView:
        if self.selected_experiment.name not in self.experiment_views:
            self.experiment_views[self.selected_experiment.name] = ExperimentRunsView(
                experiment_runs_df=get_experiment_runs_df(self.selected_experiment)
            )
        return self.experiment_views[self.selected_experiment.name]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        yield Select(
            options=[(experiment.name, experiment.id) for experiment in self.experiments],
            prompt="Select experiment"
        )
        yield ScrollableContainer(DataTable(), id="run_table")

    def on_key(self, event: Key) -> None:
        if event.key == "s":
            run_table = self.query_one(DataTable)
            cursor_coordinate = run_table.cursor_coordinate
            row_key, col_key = run_table.coordinate_to_cell_key(cursor_coordinate)

            # Filter out missing values to allow sorting
            self.experiment_view.filter_deselected_values[col_key.value].append(MISSING_VALUE)
            set_data_table_to_experiment_view(self.experiment_view, run_table)

            if self.experiment_view.current_sort is None or self.experiment_view.current_sort[0] != col_key:
                is_reverse = False
                run_table.sort(col_key, reverse=is_reverse)
            else:
                is_reverse = not self.experiment_view.current_sort[1]
                run_table.sort(col_key, reverse=is_reverse)

            self.experiment_view.current_sort = (col_key, is_reverse)
            run_table.cursor_coordinate = cursor_coordinate

        if event.key == "c":
            run_table = self.query_one(DataTable)
            cursor_coordinate = run_table.cursor_coordinate
            _, col_key = run_table.coordinate_to_cell_key(cursor_coordinate)
            self.experiment_view.deselected_columns.append(col_key.value)
            set_data_table_to_experiment_view(self.experiment_view, run_table)
            run_table.cursor_coordinate = cursor_coordinate

        if event.key == "r":
            run_table = self.query_one(DataTable)
            cursor_coordinate = run_table.cursor_coordinate
            row_key, _ = run_table.coordinate_to_cell_key(cursor_coordinate)
            self.experiment_view.deselected_run_ids.append(row_key.value)
            set_data_table_to_experiment_view(self.experiment_view, run_table)
            run_table.cursor_coordinate = cursor_coordinate

        if event.key == "f":
            run_table = self.query_one(DataTable)
            cursor_coordinate = run_table.cursor_coordinate
            row_key, col_key = run_table.coordinate_to_cell_key(cursor_coordinate)
            self.experiment_view.filter_selected_values[col_key.value] = run_table.get_cell(row_key, col_key)
            set_data_table_to_experiment_view(self.experiment_view, run_table)
            run_table.cursor_coordinate = cursor_coordinate

        if event.key == "d":
            run_table = self.query_one(DataTable)
            cursor_coordinate = run_table.cursor_coordinate
            row_key, col_key = run_table.coordinate_to_cell_key(cursor_coordinate)
            self.experiment_view.filter_deselected_values[col_key.value].append(run_table.get_cell(row_key, col_key))
            set_data_table_to_experiment_view(self.experiment_view, run_table)
            run_table.cursor_coordinate = cursor_coordinate

        if event.key == "p":
            self.experiment_view.reset_view()
            run_table = self.query_one(DataTable)
            cursor_coordinate = run_table.cursor_coordinate
            set_data_table_to_experiment_view(self.experiment_view, run_table)
            run_table.cursor_coordinate = cursor_coordinate

        if event.key == "q":
            self.quit()

    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        self.selected_experiment = [e for e in self.experiments if e.id == event.value][0]
        run_table = self.query_one(DataTable)
        set_data_table_to_experiment_view(self.experiment_view, run_table)


if __name__ == "__main__":
    app = TraktorML()
    app.run()
