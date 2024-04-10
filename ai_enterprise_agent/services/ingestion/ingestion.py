from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ai_enterprise_agent.interface.settings import ISettings
from ai_enterprise_agent.services.ingestion.csv_loader import CsvLoader
from ai_enterprise_agent.services.ingestion.msft_loader import MicrosoftLoader
from ai_enterprise_agent.services.ingestion.pdf_loader import PdfLoader
from ai_enterprise_agent.services.ingestion.txt_loader import TxtLoader


class LoaderFactory:

  @staticmethod
  def build(extension: str, config: ISettings):
    """
    This method determines the file extension of the input filename and
    creates an appropriate loader for the file based on its extension.
    """

    if extension == 'txt':  # If the file extension is 'txt', create a TxtLoader
      loader = TxtLoader(config)
    elif extension == 'pdf': # If the file extension is 'pdf', create a PdfLoader
      loader = PdfLoader(config)
    elif extension == 'csv': # If the file extension is 'csv', create a CsvLoader
      loader = CsvLoader(config)
    elif extension == 'xlsx': # If the file extension is 'xlsx', create a MicrosoftLoader
      loader = MicrosoftLoader(config)
    else:  # If the file extension is not 'txt', raise an exception
      raise Exception("Invalid loader file")

    return loader  # Return the created loader
