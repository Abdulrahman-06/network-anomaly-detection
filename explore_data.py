import pandas as pd

column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class", "difficulty_level"
]
dataframe = pd.read_csv("KDDTrain+.txt", header=None)
dataframe.columns = column_names
dataframe = pd.get_dummies(dataframe, columns=["protocol_type", "service", "flag"])
dataframe = dataframe.drop(columns=["difficulty_level"])
dataframe["is_normal"] = (dataframe["class"] != "normal").astype(int)


print(dataframe.shape)
print(dataframe.head())
print(dataframe[["class", "is_normal"]].head(10))
