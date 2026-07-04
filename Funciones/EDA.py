# funciones/eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def get_column(df):
    """
    Separa las columnas numéricas y categóricas de un DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame a analizar.

    Returns
    -------
    tuple[list, list]
        num_cols : lista de columnas numéricas.
        cat_cols : lista de columnas categóricas.
    """
    num_cols = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    cat_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return num_cols, cat_cols


def resumen_variables(df):
    """
    Muestra los posibles valores de las variables categóricas
    y estadísticas básicas de las variables numéricas.
    """

    num_cols, cat_cols = get_column(df)

    print("=" * 50)
    print("VARIABLES CATEGÓRICAS")
    print("=" * 50)

    for col in cat_cols:
        print(f"\n{col}")
        print(f"Número de categorías: {df[col].nunique(dropna=True)}")

        if df[col].nunique(dropna=True) <= 20:
            print("Valores:")
            print(df[col].dropna().unique())
        else:
            print("Primeras 20 categorías:")
            print(df[col].value_counts().head(20))

    print("\n" + "=" * 50)
    print("VARIABLES NUMÉRICAS")
    print("=" * 50)

    for col in num_cols:
        print(f"\n{col}")
        print(f"Mínimo: {df[col].min()}")
        print(f"Máximo: {df[col].max()}")
        print(f"Media: {df[col].mean():.2f}")
        print(f"Mediana: {df[col].median():.2f}")
    
    return

def distribuciones(df, num_cols):
    """
    Genera histogramas con curva de densidad.
    """

    n_cols_plot = 3
    n_rows = math.ceil(len(num_cols) / n_cols_plot)

    fig, axes = plt.subplots(
        n_rows,
        n_cols_plot,
        figsize=(16, n_rows * 4)
    )

    axes = axes.flatten()

    for i, col in enumerate(num_cols):

        sns.histplot(
            data=df,
            x=col,
            kde=True,
            ax=axes[i]
        )

        axes[i].set_title(
            f"Distribución de {col}",
            fontsize=12,
            fontweight='bold'
        )

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_boxplots(df, columnas=None, n_cols=3, figsize=(16,10)):

    if columnas is None:
        columnas = df.select_dtypes(include='number').columns

    n = len(columnas)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize
    )

    axes = axes.flatten()

    for i, col in enumerate(columnas):

        sns.boxplot(
            y=df[col],
            ax=axes[i],
            width=0.5,
            fliersize=3
        )

        axes[i].set_title(
            col,
            fontsize=12,
            fontweight='bold'
        )

        axes[i].grid(axis='x', visible=False)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def graficar_categoricas(df, cat_cols, color, top_n=20):

    n_cols = 3
    n_rows = math.ceil(len(cat_cols) / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(18, n_rows * 5)
    )


    axes = axes.flatten()

    for i, col in enumerate(cat_cols):

        conteos = df[col].value_counts().head(top_n)
        colores = sns.color_palette(color, n_colors=len(conteos))

        ax = sns.barplot(
              x=conteos.index.astype(str),
                  y=conteos.values,
                  hue=conteos.index.astype(str),
                  palette=colores,
                  legend=False,
                  ax=axes[i]
              )

        axes[i].set_title(
            f"{col}",
            fontsize=12,
            fontweight='bold'
        )

        axes[i].tick_params(
            axis='x',
            rotation=45
        )

        axes[i].set_xlabel("")
        axes[i].set_ylabel("Frecuencia")

        for p in ax.patches:
            ax.annotate(
                f"{int(p.get_height())}",
                (p.get_x() + p.get_width()/2,
                 p.get_height()),
                ha='center',
                va='bottom',
                fontsize=8
            )

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def correlacion(df):
    """
    Grafica una matriz de correlación mejorada.
    """
    corr = df.select_dtypes(include='number').corr()

    plt.figure(figsize=(12,10))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    plt.title("Matriz de correlación", fontsize=16, pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def outliers_iqr(df, num_cols):
    """
    Cuenta outliers usando IQR.
    """
    for col in num_cols:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        inferiores = q1 - 1.5 * iqr
        superiores = q3 + 1.5 * iqr

        outliers = ((df[col] < inferiores) |
                    (df[col] > superiores)).sum()

        print(f"{col}: {outliers} outliers")