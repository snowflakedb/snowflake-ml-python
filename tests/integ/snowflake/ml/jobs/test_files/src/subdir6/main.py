import yaml

with open(".config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print(config["secret_message"])
with open(".no_ext", encoding="utf-8") as f:
    content = f.read()
print(f"{content}")
