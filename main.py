from flask import Flask, request
from text_learn import TextAnalyzer
from json import dumps, load


app = Flask("analyzer")

tags = ('Hemato', 'Pediatr', 'cardio', 'derma', 'endocrino', 
        'family', 'gastro', 'gineko', 'hir', 'immuno', 
        'infectio', 'lor', 'mamolog', 'neuro', 'nevro', 
        'oftalmo', 'onco', 'procto', 'psih', 'pulmo', 
        'reab', 'rengenolog', 'stomato', 'travm', 'uro')

analyzer = TextAnalyzer("orders2", tags=tags)


@app.route('/analyze', methods=["GET"])
def analyze_text():
    text = request.args.get('text')
    current_tags = analyzer.predict(text)
    return dumps(current_tags)


@app.route("/save", methods=["GET"])
def save():
    analyzer.save()
    return "200"


@app.route("/fit", methods=["POST"])
def fit():
    data = request.json
    analyzer.set_batch_size(len(data.get('to_fit', [])))
    analyzer.set_epochs_count(data.get('epochs_count', 256))
    analyzer.set_save_after_fit(data.get('save', False))
    for frame in data.get('to_fit'):
        analyzer.fit(frame.get('text'), frame.get('tags'))


@app.route('/goFit', methods=['GET'])
def go_fit():
    with open('learnData.json', 'r') as f:
        data = load(f)
        analyzer.set_batch_size(len(data) - 1)
        analyzer.set_epochs_count(3)
        analyzer.set_save_after_fit(True)
        i = 0
        for frame in data:
            print(len(data) - i)
            i += 1
            analyzer.fit(frame.get('text'), frame.get('tags'))

app.run('0.0.0.0', 8031)

