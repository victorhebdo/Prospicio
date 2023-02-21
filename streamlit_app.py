import streamlit as st

# Define the home page
def home():
    st.write("## Welcome to the Prediction Model Website")
    st.write("This website uses a prediction model to provide information about companies.")
    st.write("")

    if st.button("Search for a Company"):
        page = "Search"
    else:
        page = "Home"

    return page

# Define the search page
def search():
    st.write("## Search for a Company")
    company_name = st.text_input("Enter a company name:")

    if st.button("Submit"):
        # Check if input is empty
        if not company_name:
            st.error("Please enter a company name.")
        else:
            # Perform prediction model here with company_name
            # Return company information as result
            st.write("## Company Information")
            st.write(f"The information for **{company_name}** is displayed below:")
            st.write("")
            st.write("### Administrative Data")
            st.write("- Address: 123 Main St.")
            st.write("- Phone: (123) 456-7890")
            st.write("- Website: https://www.example.com")
            st.write("")
            st.write("### Financial Data")
            st.write("- Revenue: $1,234,567")
            st.write("- Net Income: $123,456")

# Create the app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", ["Home", "Search"])

    if page == "Home":
        home()
    elif page == "Search":
        search()

if __name__ == "__main__":
    main()
