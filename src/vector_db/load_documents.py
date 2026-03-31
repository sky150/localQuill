from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader


# Load documents from a directory containing PDF files
def load_documents(test_data_path):
    document_loader = PyPDFDirectoryLoader(test_data_path)
    return document_loader.load()

