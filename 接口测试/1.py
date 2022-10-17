import json
import requests
import base64
url = "http://localhost:3000/api/tasks"
headers = {"keyId": "4xxxx-b01c-7ef0801c8d4f", "BusinessCode": "xxx", "Content-Type":"application/json;charset=utf-8" }
body = {
    "XXXQUEST": {
        "PARAMDATA": {
            "vehicleNo": "XX2005",
            "startPostime": "2018-01-12 12:00:00",
            "endPostime": "2018-01-12 12:02:00"
        }
    }
}

s = requests.session()
r = s.post(url, headers=headers, data=json.dumps(body), verify=False)  # verify参数来解决ssl报错问题
print("*"*10, "返回结果", "*"*10)
print(r.text)
print("*"*10, "Base64解密结果", "*"*10)
rs = base64.b64decode(r.text)
print(rs)
# print(type(rs))
print("\xe6\x88\x90\xe5\x8a\x9f".encode('raw_unicode_escape').decode())