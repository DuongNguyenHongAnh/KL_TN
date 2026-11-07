# Build script for KLTN Word Document
import os, sys, docx, pandas as pd
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'experiment_results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
OUTPUT_FILE = os.path.join(BASE_DIR, 'Khoa_Luan_Tot_Nghiep.docx')
DRAFT_FILE = r'E:\KLTN\Du_thao_KLTN_final.docx'

print('Script builder part 1 written')
