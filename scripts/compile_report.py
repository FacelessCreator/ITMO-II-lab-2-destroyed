import os

REPORT_TEMPLATE_FILEPATH = 'src/report_template.md'
REPORT_FOLDER = 'build'
REPORT_FILENAME = 'report.md'
TRAIN_GRAPHICS_FOLDER = 'train_graphics'

report_template_file = open(REPORT_TEMPLATE_FILEPATH)
report_template = ''
for line in report_template_file:
    report_template+=line

report_template_file.close()

graphic_descriptions = ''

for filename in os.listdir(REPORT_FOLDER+'/'+TRAIN_GRAPHICS_FOLDER):
    filepath = TRAIN_GRAPHICS_FOLDER+'/'+filename
    filename_noextension = os.path.splitext(filename)[0]
    graphic_name = filename_noextension.replace('_',' ')
    graphic_descriptions += '### {}\n\n![{}]({})\n\nЗдесь нужно прокомментировать график\n\n'.format(graphic_name, graphic_name, filepath)

report = report_template.replace('GRAPHICS_HERE', graphic_descriptions)

report_file = open(REPORT_FOLDER+'/'+REPORT_FILENAME, 'w')
report_file.write(report)
report_file.close()
