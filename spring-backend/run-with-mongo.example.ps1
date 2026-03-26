# Copy to run-with-mongo.ps1, paste your URI, then:
#   .\run-with-mongo.ps1
$env:MONGODB_URI = "mongodb+srv://USER:PASSWORD@cluster0.enwyvnr.mongodb.net/Stress_Dtabase?retryWrites=true&w=majority"
Set-Location $PSScriptRoot
mvn spring-boot:run
