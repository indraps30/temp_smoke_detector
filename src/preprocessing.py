# Import required libraries.
import pandas as pd
import util as utils
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

def load_dataset(config: dict):
    """Load and concat the train, valid, and test set."""
    # Load the train, valid, test sets.
    X_train = utils.pickle_load(config["train_set_path"][0])
    y_train = utils.pickle_load(config["train_set_path"][1])
    
    X_valid = utils.pickle_load(config["valid_set_path"][0])
    y_valid = utils.pickle_load(config["valid_set_path"][1])
    
    X_test = utils.pickle_load(config["test_set_path"][0])
    y_test = utils.pickle_load(config["valid_set_path"][1])
    
    # Concat the dataset.
    train_set = pd.concat([X_train, y_train], axis = 1)
    valid_set = pd.concat([X_valid, y_valid], axis = 1)
    test_set = pd.concat([X_test, y_test], axis = 1)
    
    # Return 3 set of data.
    return train_set, valid_set, test_set

def rus_fit_resample(set_data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Random under sampling the dataset."""
    # Create copy of set data.
    set_data = set_data.copy()
    
    # Create sampling object.
    rus = RandomUnderSampler(random_state = 42)
    
    # Balancing set data.
    result = rus.fit_resample(
        set_data.drop(columns = config["label"]),
        set_data[config["label"]]
    )
    X_rus, y_rus = result[:2]    
    
    # Concatenate balanced data.
    set_data_rus = pd.concat([X_rus, y_rus], axis = 1)
    
    # Return balanced data.
    return set_data_rus

def ros_fit_resample(set_data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Random over sampling the dataset."""
    # Create copy of set data.
    set_data = set_data.copy()
    
    # Create sampling object.
    ros = RandomOverSampler(random_state = 42)
    
    # Balancing set data.
    result = ros.fit_resample(
        set_data.drop(columns = config["label"]),
        set_data[config["label"]]
    )
    X_ros, y_ros = result[:2]    
    
    # Concatenate balanced data.
    set_data_ros = pd.concat(
        [X_ros, y_ros],
        axis = 1
    )
    
    # Return balanced data.
    return set_data_ros

def sm_fit_resample(set_data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """SMOTE sampling the dataset."""
    # Create copy of set data.
    set_data = set_data.copy()
    
    # Create sampling object.
    sm = SMOTE(random_state = 42)
    
    # Balancing set data.
    result = sm.fit_resample(
        set_data.drop(columns = config["label"]),
        set_data[config["label"]]
    )
    X_sm, y_sm = result[:2]    
    
    # Concatenate balanced data.
    set_data_sm = pd.concat([X_sm, y_sm], axis = 1)
    
    # Return balanced data.
    return set_data_sm

def remove_outliers(set_data):
    """Remove outliers using IQR method."""
    # Create copy set of data.
    set_data = set_data.copy()
    
    list_of_set_data = list()
    
    # Do the outliers handling.
    for col_name in set_data.columns[:-1]:
        q1 = set_data[col_name].quantile(0.25)
        q3 = set_data[col_name].quantile(0.75)
        iqr = q3 - q1
        set_data_cleaned = set_data[~((set_data[col_name] < (q1 - 1.5 * iqr)) | (set_data[col_name] > (q3 + 1.5 * iqr)))].copy()
        list_of_set_data.append(set_data_cleaned.copy())
        
    # Concatenate the cleaned dataset.
    set_data_cleaned = pd.concat(list_of_set_data)
    count_duplicated_index = set_data_cleaned.index.value_counts()
    used_index_data = count_duplicated_index[count_duplicated_index == (set_data.shape[1]-1)].index
    set_data_cleaned = set_data_cleaned.loc[used_index_data].drop_duplicates()
    
    # Return the cleaned dataset.
    return set_data_cleaned


if __name__ == "__main__":
    # 1. Load configuration file.
    config = utils.load_config()
    
    # 2. Load dataset.
    train_set, valid_set, test_set = load_dataset(config)
    
    # 3. Undersampling dataset.
    train_set_rus = rus_fit_resample(train_set, config)
    
    # 4. Oversampling dataset.
    train_set_ros = ros_fit_resample(train_set, config)
    
    # 5. SMOTE dataset.
    train_set_sm = sm_fit_resample(train_set, config)
    
    # 6. Removing outliers.
    train_set_rus_cleaned = remove_outliers(train_set_rus)
    train_set_ros_cleaned = remove_outliers(train_set_ros)
    train_set_sm_cleaned = remove_outliers(train_set_sm)
    
    # 7. Dump dataset.
    X_train = {
        "Undersampling" : train_set_rus_cleaned[config["predictors"]],
        "Oversampling" : train_set_ros_cleaned[config["predictors"]],
        "SMOTE" : train_set_sm_cleaned[config["predictors"]]
    }
    
    y_train = {
        "Undersampling" : train_set_rus_cleaned[config["label"]],
        "Oversampling" : train_set_ros_cleaned[config["label"]],
        "SMOTE" : train_set_sm_cleaned[config["label"]]
    }
    
    utils.pickle_dump(
        X_train,
        config["train_feng_set_path"][0]
    )
    
    utils.pickle_dump(
        y_train,
        config["train_feng_set_path"][1]
    )
    
    utils.pickle_dump(
        valid_set[config["predictors"]],
        config["valid_feng_set_path"][0]
    )
    
    utils.pickle_dump(
        valid_set[config["label"]],
        config["valid_feng_set_path"][1]
    )
    
    utils.pickle_dump(
        test_set[config["predictors"]],
        config["test_feng_set_path"][0]
    )
    
    utils.pickle_dump(
        test_set[config["label"]],
        config["test_feng_set_path"][1]
    )