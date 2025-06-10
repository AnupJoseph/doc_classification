import pandas as pd
import plotly.express as px


def distribution_plot(df: pd.DataFrame, id2label, label_name="label"):
    counts = pd.DataFrame(df[label_name].value_counts())
    counts.reset_index(inplace=True)
    counts["label_name"] = counts[label_name].apply(lambda x: id2label[x])
    counts = counts[["count", "label_name"]]
    counts.set_index("label_name", inplace=True)
    return px.bar(counts,labels={"value":"Count","label_name":"Label"},title="Data distribution")
