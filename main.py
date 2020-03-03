from pdf.PdfExport import PdfExport
from model.Model import Model

if __name__ == '__main__':
    PdfExport.export_data()
    model = Model()
    model.train("5")
