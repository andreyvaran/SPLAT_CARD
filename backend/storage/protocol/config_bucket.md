# Политики доступа (Bucket Policy)

Чтобы создать bucket, который будет работать со всеми разрешениями и с правильной политикой для нашего сервера нужно 

1) Скопировать политику у одного из бакетов, которые уже есть и работают командой:

``aws s3api get-bucket-policy --endpoint-url=https://s3.cloud.ru  --bucket {Название бакета}``

2) Дальше применить эту политику для нового бакета 
`` aws s3api put-bucket-policy --bucket --endpoint-url=https://s3.cloud.ru test --policy '{нужная нам политика}'``


Вот политика, которая подходит для работы 

'{
  "Version":"2012-10-17",
  "Statement":[
    {          
      "Sid":"",        
      "Effect":"Allow",
      "Principal":"*",  
      "NotPrincipal":{},      
      "Action":"s3:GetObject",
      "NotAction":[],                             
      "Resource":"arn:aws:s3:::test/*",           
      "NotResource":[],
      "Condition":{}
    }
  ]
}'

Документация:
https://cloud.ru/docs/s3e/ug/topics/security__bucketpolicy.html

Примеры команд:

1) aws s3api get-bucket-policy --endpoint-url=https://s3.cloud.ru  --bucket story-content


2)  aws s3api put-bucket-policy --bucket --endpoint-url=https://s3.cloud.ru test --policy '{
  "Version":"2012-10-17",
  "Statement":[
    {          
      "Sid":"",        
      "Effect":"Allow",
      "Principal":"*",  
      "NotPrincipal":{},      
      "Action":"s3:GetObject",
      "NotAction":[],                             
      "Resource":"arn:aws:s3:::test/*",           
      "NotResource":[],
      "Condition":{}
    }
  ]
}'
