import httpx

# Define the base url for the API.
base_url = "http://localhost:5000" # Replace with your API's URL
client = httpx.Client(base_url=base_url)

def login(username, password, client=client):
    data = {"username": username, "password": password, "grant_type": "password"}
    response = client.post(f"{base_url}/login", data=data)
    if response.status_code != 200:
        raise Exception("Failed to authenticate.")
    print(response.json())


def whoami(client=client):
    headers = {"Authorization": f"Bearer {token}"}
    response = httpx.get(f"{base_url}/users/whoami", headers=headers)
    return response.json()


def get_client():
    login("123", "123")
    return client