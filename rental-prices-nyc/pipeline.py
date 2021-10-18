from logging import Logger
from clearml.automation.controller import PipelineDecorator
from clearml import Task


@PipelineDecorator.component(return_values=["dataset_id"], cache=True, task_type=Task.TaskTypes.data_processing)
def create_dataset(project_name: str, dataset_name: str) -> str:
    from clearml import StorageManager, Dataset, Logger
    import pandas as pd
    import plotly.express as px

    try:
        original_ds = Dataset.get(dataset_project=project_name, dataset_name=dataset_name, dataset_tags=["latest"])
        Logger.current_logger().current_logger().report_text("Dataset {} already exists, creating the child one".format(dataset_name))
        ds = Dataset.create(dataset_name=dataset_name, parent_datasets=[original_ds.id])
    except ValueError:
        original_ds = None
        Logger.current_logger().current_logger().report_text("Dataset {} does not exists, creating the first one".format(dataset_name))
        ds = Dataset.create(dataset_project=project_name, dataset_name=dataset_name)

    Logger.current_logger().current_logger().report_text("Downloading AirBnb NYC 2019 dataset")
    dataset_csv = StorageManager.get_local_copy(
        remote_url="https://raw.githubusercontent.com/adishourya/Airbnb/master/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
        
    ds.add_files(dataset_csv)
    ds.upload()
    ds.finalize()
    ds.tags = []
    ds.tags = ["latest"]
    if original_ds:
        original_ds.tags = []

    df = pd.read_csv(dataset_csv)
    fig = px.histogram(df, x="price", marginal="box")
    Logger.current_logger().current_logger().report_plotly(title="TOTAL Price distribution", series="Price", iteration=0, figure=fig)

    fig = px.histogram(df, x="price", marginal="box", color="neighbourhood_group")
    Logger.current_logger().current_logger().report_plotly(title="Price distribution by Neighbourhood", series="Price", iteration=0, figure=fig)
    
    dataset_id = ds.id
    return dataset_id


@PipelineDecorator.component(return_values=["dataset_id"], cache=True, task_type=Task.TaskTypes.data_processing)
def clean_data(project_name: str, dataset_id: str, dataset_name_cleaned: str, min_price: int, max_price: int) -> str:
    from clearml import Dataset, Logger
    import pandas as pd
    import glob
    import plotly.express as px

    #bug workarounf in clearml pipelines
    min_price = int(min_price)
    max_price = int(max_price)

    dataset_cleaned_name = "dataset.csv"

    ds_raw = Dataset.get(dataset_id=dataset_id, dataset_tags=["latest"])
    csv_files = glob.glob("%s/*.csv" % ds_raw.get_local_copy())
    df = pd.concat(map(pd.read_csv, csv_files), ignore_index=True)

    idx = df["price"].between(min_price, max_price)
    df = df[idx].copy()
    Logger.current_logger().current_logger().report_text("Dataset price outliers removal outside range: {}-{}".format(min_price, max_price))
    df["last_review"] = pd.to_datetime(df["last_review"])
    Logger.current_logger().current_logger().report_text("Dataset last_review data type fix")

    idx = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
    df = df[idx].copy()

    df.to_csv(dataset_cleaned_name, index=False)

    try:
        original_ds = Dataset.get(dataset_name=dataset_name_cleaned, dataset_tags=["latest"])
        Logger.current_logger().current_logger().report_text("Dataset {} already exists, creating the child one".format(dataset_name_cleaned))
        ds = Dataset.create(dataset_project=project_name, dataset_name=dataset_name_cleaned, parent_datasets=[original_ds])
    except ValueError:
        original_ds = None
        Logger.current_logger().current_logger().report_text("Dataset {} does not exists, creating the first one".format(dataset_name_cleaned))
        ds = Dataset.create(dataset_project=project_name, dataset_name=dataset_name_cleaned)
    ds.add_files(dataset_cleaned_name)
    ds.upload()
    ds.finalize()
    ds.tags = []
    ds.tags = ["latest"]
    if original_ds:
        original_ds.tags = []

    fig = px.histogram(df, x="price", marginal="box")
    Logger.current_logger().current_logger().report_plotly(title="TOTAL Price distribution", series="Price", iteration=0, figure=fig)

    fig = px.histogram(df, x="price", marginal="box", color="neighbourhood_group")
    Logger.current_logger().current_logger().report_plotly(title="Price distribution by Neighbourhood", series="Price", iteration=0, figure=fig)

    dataset_id = ds.id
    return dataset_id


@PipelineDecorator.component(return_values=["dataset_id"], cache=True, task_type=Task.TaskTypes.data_processing)
def check_data(dataset_id: str, min_price: int, max_price: int) -> str:
    from clearml import Dataset, Logger
    import pandas as pd
    import numpy as np
    import glob
    from pandas_profiling import ProfileReport

    #bug workarounf in clearml pipelines
    min_price = int(min_price)
    max_price = int(max_price)

    ds = Dataset.get(dataset_id=dataset_id, dataset_tags=["latest"])
    csv_files = glob.glob("%s/*.csv" % ds.get_local_copy())
    df = pd.concat(map(pd.read_csv, csv_files), ignore_index=True)

    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = df.columns.values
    Logger.current_logger().report_text("Test column names: column names expected: {}".format(list(expected_colums)))
    Logger.current_logger().report_text("Test column names: column names in dataset: %{}".format(list(these_columns)))
    assert list(expected_colums) == list(these_columns)

    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
    neigh = set(df["neighbourhood_group"].unique())
    Logger.current_logger().report_text("Test column names: neighbourhood group expected: {}".format(set(known_names)))
    Logger.current_logger().report_text("Test column names: neighbourhood group in dataset: {}".format(set(neigh)))
    assert set(known_names) == set(neigh)

    idx = df["longitude"].between(-74.25, - 73.50) & df["latitude"].between(40.5, 41.2)
    Logger.current_logger().report_text("Test proper boundaries: unexpected items are {}".format(np.sum(~idx)))
    assert np.sum(~idx) == 0

    Logger.current_logger().report_text("Test row count: items are {}".format(df.shape[0]))
    assert 15000 < df.shape[0] < 1000000

    items_ok = df["price"].between(min_price, max_price).shape[0]
    Logger.current_logger().report_text("Price range test, items in range are: {}".format(items_ok))
    assert df.shape[0] == items_ok

    profile = ProfileReport(df)
    profile.to_file("pp-output.html")
    Logger.current_logger().report_media("html", "pandasprofiling", local_path="pp-output.html")

    dataset_id = ds.id
    return dataset_id


@PipelineDecorator.component(return_values=["processed_data"], cache=True, task_type=Task.TaskTypes.data_processing)
def train_model(dataset_id: str):
    from clearml import Dataset, Logger
    import pandas as pd
    import glob
    from sklearn.model_selection import StratifiedKFold


    ds = Dataset.get(dataset_id=dataset_id, dataset_tags=["latest"])
    csv_files = glob.glob("%s/*.csv" % ds.get_local_copy())
    df = pd.concat(map(pd.read_csv, csv_files), ignore_index=True)
    
    skf = StratifiedKFold(n_splits=5)
    target = df.loc[:,"neighbourhood_group"]

    logger = Logger.current_logger()
    logger.report_text("Start training")

    fold_no = 1
    for train_index, test_index in skf.split(df, target):
        train = df.loc[train_index,:]
        test = df.loc[test_index,:]
        logger.report_text("Fold: {} Class ratio: {}".format(fold_no, sum(test["Returned_Units"])/len(test["Returned_Units"])))
        fold_no += 1


@PipelineDecorator.pipeline(name="Airbnb nyc model pipeline", project="examples", version="0.0.1")
def executing_pipeline(project_name: str, dataset_name_raw: str, dataset_name_cleaned: str, min_price: int=10, max_price: int=350):
    raw_dataset_id = create_dataset(project_name, dataset_name_raw)
    cleaned_dataset_id = clean_data(project_name, raw_dataset_id, dataset_name_cleaned, min_price, max_price)
    checked_dataset_id = check_data(cleaned_dataset_id, min_price, max_price)
    train_model(checked_dataset_id)


if __name__ == "__main__":
    PipelineDecorator.set_default_execution_queue("default")
    # PipelineDecorator.debug_pipeline()

    executing_pipeline(
        project_name="examples",
        dataset_name_raw="airbnb-nyc-2019-raw",
        dataset_name_cleaned="airbnb-nyc-2019-cleaned",
        min_price=10,
        max_price=350,
    )
