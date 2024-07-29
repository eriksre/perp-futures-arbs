import requests

class Webhook:
    def __init__(self, webhook_type='testnet'):
        self.webhook_urls = {
            'testnet': 'YOUR_TESTNET_WEBHOOK_URL_HERE',
            'mainnet': 'YOUR_MAINNET_WEBHOOK_URL_HERE'
        }
        
        self.webhook_url = self.webhook_urls.get(webhook_type, self.webhook_urls['testnet'])

    def send_message(self, content):
        data = {"content": content}
        response = requests.post(self.webhook_url, json=data)
        if response.status_code != 204:
            print(f"Failed to send message: {response.status_code}, {response.text}")
        else:
            print("Message sent successfully.")


#example usage
# # For testnet
# testnet_webhook = Webhook(webhook_type='testnet')
# testnet_webhook.send_message(message)

# # For mainnet
# mainnet_webhook = Webhook(webhook_type='mainnet')
# mainnet_webhook.send_message(message)

# # For futures
# futures_webhook = Webhook(webhook_type='futures')
# futures_webhook.send_message(message)
