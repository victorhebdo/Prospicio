import base64
import json
import os
import io

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import streamlit as st
from st_aggrid import AgGrid

import config
from generate_products import LisiProducts
from ocr_bounding_box import OCR, get_bounding_box
from utils import convert_df_csv, convert_df_json


def page_login(auth_status: bool, name: str = None):
    if auth_status:
        st.write("You are logged in.")
        st.write(f"Welcome {name} !")
    elif not auth_status:
        st.error("Username/password is incorrect")
    elif auth_status is None:
        st.warning("Please enter your username and password")


def page_input_regex():
    if st.session_state.get("references_df", None) is not None:
        st.warning("A table of references has already been computed")

    st.title("Input RegEx")

    regex_input_raw = st.text_area("Enter the RegEx of the Standard here", config.DEFAULT_REGEX)

    regex_input = regex_input_raw.replace("\n", "").replace(" ", "")
    fields = [x.strip("), \n") for x in regex_input.split("(")][1:]

    index_mat = index_definition("Material", fields)
    index_coat = index_definition("Coating", fields)
    index_diam = index_definition("Diameter", fields)
    index_over = index_definition("Oversize", fields)
    # index_len = index_definition("Length", fields)  # DOES NOT INFLUENCE ANYTHING AT THIS POINT
    st.info("The correspondence between codes and values is specified in the CSV below ")

    st.title("Form - input correspondence")

    product_name = st.text_input("Piece name", f"{fields[0]}")
    # product_finish = st.selectbox("Piece finish", options=config_OCR.CONFIG_ITGR["digit_12"].keys()) # ITGR FIRST LETTERS
    product_type = st.selectbox("Product type", options=config.TYPES_LIST)

    product_emitter = st.selectbox(
        "Emitter company", options=config.STANDARD_EMITTERS["emitter_list"]
    )

    (
        df_mat_codes,
        mat_fixed_str,
        df_diam_codes,
        diam_fixed_str,
        df_oversize_codes,
    ) = input_diam_mat_over()

    if st.button("COMPUTE REFERENCES"):
        lisi_product = LisiProducts(
            product_name=product_name,
            regex=regex_input,
            prod_type=product_type,
            fixed_diam=diam_fixed_str,
            fixed_material=mat_fixed_str,
            index_material=index_mat,
            index_coating=index_coat,
            index_diameter=index_diam,
            index_oversize=index_over,
            regex_constraints="",
            diameter_df=df_diam_codes,
            oversize_df=df_oversize_codes,
            material_df=df_mat_codes,
            standard_emitter=product_emitter,
        )

        df_references = lisi_product.get_products_data()
        df_references['Other'] = ""  # Client's wish to have a "divers" column in the exported file
        st.success("The references were generated successfully ! Go to next page")
        st.session_state["references_df"] = df_references
        st.session_state["product_instance"] = lisi_product


def page_input_ocr():
    if st.session_state.get("references_df", None) is not None:
        st.warning("A table of references has already been computed")

    st.title("Show Standard - compute OCR - WIP")

    uploaded_standard = st.file_uploader(
        "Upload the Standard or Plan (as a .pdf, .png, .jpg)",
        type=["pdf", "png", "jpg"],
    )
    if uploaded_standard is not None:
        standard_check_display(uploaded_standard)

    ocr_computation()

    st.title("Write RegEx fields")

    regex_name = st.text_input(
        "The name of the standard/product", "HST13"
    )

    regex_fields_mat = regexfield_definition("Material")
    regex_fields_coat = regexfield_definition("Coating")
    regex_fields_diam = regexfield_definition("Diameter")
    regex_fields_over = regexfield_definition("Length")
    regex_fields_over = regexfield_definition("Oversize")

    st.text_area("The defined RegEx is : ",
                 "".join([regex_name, regex_fields_mat, regex_fields_coat, regex_fields_diam, regex_fields_over])
    )

    if st.button("VALIDATE Regex"):
        st.warning("This yet has to be developed. WIP")
    # index_len = index_definition("Length", fields)  # DOES NOT INFLUENCE ANYTHING AT THIS POINT
    st.info(
        "The correspondence between codes and values is specified in the CSV below "
    )

    st.title("Form - input correspondences")

    if st.button("MAGIC button : PRE-FILL these fields"):
        st.warning("This yet has to be developed. WIP")

    product_name = regex_name
    # product_finish = st.selectbox("Piece finish", options=config_OCR.CONFIG_ITGR["digit_12"].keys()) # ITGR FIRST LETTERS
    product_type = st.selectbox("Product type", options=config.TYPES_LIST)

    product_emitter = st.selectbox(
        "Emitter company", options=config.STANDARD_EMITTERS["emitter_list"]
    )

    (
        df_mat_codes,
        mat_fixed_str,
        df_diam_codes,
        diam_fixed_str,
        df_oversize_codes,
    ) = input_diam_mat_over()

    if st.button("COMPUTE REFERENCES"):
        st.warning("So far, you have to compute refs from 1A page. WIP")


def ocr_computation():
    """
    Launch OCR with boundingbox class and config_OCR
    :return:
    """
    os.environ["TESSDATA_PREFIX"] = "C:/Program Files/Tesseract-OCR/tessdata"

    use_case = st.selectbox(
        "Select a usecase for OCR recognition",
        ["HST13", "EN6115", "ST2135", "U633A5250501"],
    )
    path_pdf = config.PDF_STANDARD[use_case]

    CAT_OCR = ["company", "material", "piece_type", "important", "other"]
    categories_ocr = st.multiselect(label="Categories to lookup with OCR",
                                    options=CAT_OCR,
                                    default=CAT_OCR)

    run_ocr = st.button("Run OCR")
    if run_ocr:
        ocr = OCR(
            config_path=config.CONFIG_WORDS,
            pdf_path=path_pdf,
            visualisation_config_path=config.CONFIG_VISU,
            poppler_path=config.POPPLER_PATH,
        )

        ocr.adjust_words_lookup(category_list=categories_ocr)
        st.markdown(ocr.display_color_legend(categories_ocr), unsafe_allow_html=True)

        for i, page in enumerate(ocr.pages):
            results = get_bounding_box(page)
            img = ocr.display_bb_confidence(results, page)
            ocr.add_bb_on_pages()
            st.image(image=img, caption=f"Page: {str(i + 1)}")


def page_references(df_show: pd.DataFrame = None, regex_input: str = None):
    st.title("Generated references")
    if df_show is None:
        st.text("The references were not yet generated, go to input page.")
    else:
        st.text(
            f"There are {df_show.shape[0]} references generated, among which XXX are already in M3 database."
        )
        st.success(
            "Here are the generated references (you can explore and apply filters on the column values) : "
        )
        AgGrid(df_show)

        st.text(f"Possible ITCL are : {df_show.ITCL.unique()}")
        st.text(f"Possible ITGR are : {df_show.ITGR.unique()}")
        st.text(f"Possible assignations are : {set([it for sublist in df_show.CFI1.to_list() for it in sublist])}")

        st.title("Export the references")
        nameproduct = st.session_state['product_instance'].product_name
        timeproduct = st.session_state['product_instance'].date_time
        st.download_button(
            label="Download CSV (12 columns) for BDT",
            data=convert_df_csv(df_show),
            file_name=f"references_{nameproduct}_{timeproduct}.csv",
            mime="text/csv",
        )

        if st.button("Export directly to BDT"):
            st.text("This does nothing so far. WIP")

        config_to_export = export_product_config(st.session_state["product_instance"])
        st.download_button(
            label="Export the product config used to generate refs",
            data=config_to_export,
            file_name=f"config_{nameproduct}_{timeproduct}.json",
            mime="text/json",
        )


def page_analysis(df_analysis):
    st.header("Analysis of the references")

    if df_analysis is None:
        st.text("The references were not yet generated, go to input page.")

    else:
        st.info(
            f"There are {df_analysis.shape[0]} references generated for the RegEx."
        )
        st.text(f"Possible ITCL are : {df_analysis.ITCL.unique()}")
        st.text(f"Possible ITGR are : {df_analysis.ITGR.unique()}")
        st.text(f"Possible assignations are : {set([it for sublist in df_analysis.CFI1.to_list()for it in sublist])}")

        col1 = st.text_input("Column to show", "ITCL")
        countplot(df_analysis, col1)

        col2 = st.text_input("Column to show", "ITGR")
        countplot(df_analysis, col2)

        col3 = st.text_input("Column to show", "CFI1")
        fig = plt.figure()
        plt.title(f"Plotting {col3} counts")
        pd.Series([it for sublist in df_analysis[col3].to_list() for it in sublist]).value_counts().plot(kind="bar")
        st.pyplot(fig)


def input_diam_mat_over():
    """
    To input the diameter and material, with checkbox if fixed or from csv
    :return:
    """
    df_diam_codes = None
    diam_fixed_str = None
    df_oversize_codes = None
    df_mat_codes = None
    mat_fixed_str = None

    if st.checkbox("Single diameter"):
        diam_fixed_str = float(st.number_input("Enter diameter (mm)", 0.0, 100.0))
    else:
        diam_csv = st.file_uploader("Diameter code table (CSV, sep=';')", type="csv")
        if diam_csv is not None:
            df_diam_codes = pd.read_csv(
                io.StringIO(diam_csv.read().decode("utf-8")), sep=";"
            )

    if st.checkbox("Oversize possible ?"):
        oversize_csv = st.file_uploader("Oversize code table (CSV, sep=';')", type="csv")
        if oversize_csv is not None:
            df_oversize_codes = pd.read_csv(
                io.StringIO(oversize_csv.read().decode("utf-8")), sep=";"
            )

    if st.checkbox("Single material ?"):
        mat_fixed_str = st.selectbox("Enter material", options=config.MATERIALS_LIST)
    else:
        material_csv = st.file_uploader("Material code table (CSV, sep=';')", type="csv")
        if material_csv is not None:
            df_mat_codes = pd.read_csv(
                io.StringIO(material_csv.read().decode("utf-8")), sep=";"
            )

    return df_mat_codes, mat_fixed_str, df_diam_codes, diam_fixed_str, df_oversize_codes


def index_definition(cat_name: str, fields_list: list) -> int:
    index = -1
    if st.checkbox(f"{cat_name}  specified in regex ?"):
        index = int(st.number_input(f"{cat_name} position", -1, 5))
        if index != -1:
            possibilities_mat = fields_list[index].split("|")
            st.text(f"{cat_name} possibilities : {possibilities_mat}")
    return index


def regexfield_definition(cat_name: str, ) -> int:
    field_possibilities = ""
    DEFAULT_FIELDS = "(AB|AG|AP|AZ|BJ|CT|HK|GD|GM|K|MA|RP|RS|RV|SU|TB|UV|VF|WF|YW|NKA|NKB|NKC|NKJ|NKK|NKL|NAP|-)"
    if st.checkbox(f"{cat_name}  specified in regex ?"):
        field_possibilities = st.text_input(f"{cat_name} possibilities :", DEFAULT_FIELDS)
    return field_possibilities


def export_product_config(product_instance):
    attribute_dict = vars(product_instance)
    for df in ["material_df", "diameter_df", "oversize_df"]:
        if type(attribute_dict[df]) == pd.DataFrame:
            new_value = attribute_dict[df].to_dict()
            attribute_dict[df] = new_value
    config_to_export = json.dumps(attribute_dict, indent=2)
    return config_to_export


def export_product_config_parser_like(product_instance):
    attribute_dict = vars(product_instance)
    for df in ["material_df", "diameter_df", "oversize_df"]:
        if type(attribute_dict[df]) == pd.DataFrame:
            new_value = attribute_dict[df].to_dict()
            attribute_dict[df] = new_value
    config_to_export = json.dumps(attribute_dict)
    return config_to_export


def countplot(df, category):
    """
    Plot a categorical countplot with seaborn in streamlit
    """
    fig = plt.figure(figsize=(10, 4))
    sns.countplot(x=category, data=df)
    plt.semilogy()
    plt.ylim(bottom=1)
    st.pyplot(fig)


def countplot_series(df):
    """
    Plot a categorical countplot with seaborn in streamlit
    """
    fig = plt.figure(figsize=(10, 4))
    sns.countplot(data=df)
    plt.semilogy()
    st.pyplot(fig)


def standard_check_display(uploaded_standard):
    # To read file as bytes:
    if uploaded_standard.name.split(".")[1] in ["png", "jpg"]:
        bytes_data = uploaded_standard.getvalue()
        st.image(bytes_data)
    elif uploaded_standard.name.split(".")[1] in ["pdf"]:
        display_pdf(uploaded_standard)
    else:
        st.text("File is in invalid format (not jpg, png or pdf).")


def display_pdf(uploaded_file):
    # Opening file from file path
    base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
    # Solution 2
    pdf_display = (
        f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
        f'width="900" height="500" type="application/pdf"></iframe>'
    )
    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)
