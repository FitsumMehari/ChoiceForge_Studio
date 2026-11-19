"""Install royalty-free Kokoro speech model (one-time 136 MB download)."""
from free_voice import RELEASE, install_models, model_status

if __name__ == '__main__':
    print('Downloading a free speech model from:', RELEASE)
    print('No accounts, API keys, or paid subscriptions are needed.')
    try:
        files = install_models()
    except Exception as exc:
        raise SystemExit('SETUP FAILED: ' + str(exc)) from exc
    print('Downloaded:', ', '.join(files) if files else 'nothing; files already installed')
    print('Verified:', model_status())
