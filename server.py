#!flask/bin/python
import io

from flask import Flask, render_template, request, send_file
# from gevent.pywsgi import WSGIServer
from TTS.utils.synthesizer import Synthesizer
from synthesizer_waveglow import Synthesizer as WGSynthesizer

tts_checkpoint = 'models/TTS_model.pth.tar'
tts_config = 'models/config.json'
vocoder_checkpoint= 'models/checkpoint_1450000.pth.tar'
vocoder_config = 'models/v_config.json'
my_vocoder_melgan= 'models/myvocoder.pth.tar'
my_vocoder_checkpoint= 'models/waveglow_1070000'
my_vocoder_config = 'models/myv_config.json'
synthesizer = Synthesizer(tts_checkpoint, tts_config, vocoder_checkpoint, vocoder_config, False, False)
synthesizers = Synthesizer(tts_checkpoint, tts_config, my_vocoder_melgan, vocoder_config, False, False)
synth = WGSynthesizer(tts_checkpoint, tts_config, my_vocoder_checkpoint, my_vocoder_config, False, True)

app = Flask(__name__)


@app.route('/')
def indexv2():
    return render_template('indexv2.html', show_details=False)

# @app.route('/details')
# def details():
#     model_config = load_config('models/config.json')
#     # if args.vocoder_config is not None and os.path.isfile(args.vocoder_config):
#     #     vocoder_config = load_config(vocoder_config)
#     # else:
#     #     vocoder_config = None
#
#     return render_template('details.html',
#                            show_details=args.show_details
#                            , model_config=model_config
#                            , vocoder_config=vocoder_config
#                            , args=args.__dict__
#                           )

@app.route('/api/tts', methods=['GET'])
def tts():
    text = request.args.get('text')
    print(" > Model input: {}".format(text))
    wavs = synthesizer.tts(text)
    out = io.BytesIO()
    synthesizer.save_wav(wavs, out)
    return send_file(out, mimetype='audio/wav')

@app.route('/api/vocoder', methods=['GET'])
def vocoder():
    text = request.args.get('text')
    print(" > Model input: {}".format(text))
    wavs = synthesizers.tts(text)
    out = io.BytesIO()
    synthesizers.save_wav(wavs, out)
    return send_file(out, mimetype='audio/wav')


# This will load WAVEGLOW VOCODER
@app.route('/api/wgvocoder', methods=['GET'])
def wavvocoder():
    text = request.args.get('text')
    print(" > Model input: {}".format(text))
    wavs = synth.tts(text)
    out = io.BytesIO()
    synth.save_wav(wavs, out)
    return send_file(out, mimetype='audio/wav')

def main():
    app.run(debug=False, host='0.0.0.0', port=5002)
    # http_server = WSGIServer(('', 5000), app)
    # http_server.serve_forever()

if __name__ == '__main__':
    main()
