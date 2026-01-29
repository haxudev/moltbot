# Deploying Moltbot to Azure Web App for Containers

This guide covers deploying Moltbot gateway to Azure Web App for Containers with Azure OpenAI as the LLM provider.

## Prerequisites

- Azure CLI installed and authenticated (`az login`)
- Docker (for local testing, optional)
- An Azure subscription with permissions to create resources

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Azure Web App                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Container (Node.js)                     │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │           Moltbot Gateway                    │    │   │
│  │  │  - Control UI (WebSocket)                    │    │   │
│  │  │  - Agent runtime                             │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Azure OpenAI                             │
│  - GPT-5.2, GPT-5.2-codex, etc.                               │
└─────────────────────────────────────────────────────────────┘
```

## Step 1: Create Azure Resources

### 1.1 Create Resource Group

```bash
RESOURCE_GROUP="moltbot-rg"
LOCATION="eastus2"

az group create --name $RESOURCE_GROUP --location $LOCATION
```

### 1.2 Create Azure Container Registry (ACR)

```bash
ACR_NAME="moltbotreg"  # Must be globally unique

az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Basic \
  --admin-enabled true
```

### 1.3 Create Azure OpenAI Resource

```bash
OPENAI_NAME="moltbot-openai"

az cognitiveservices account create \
  --name $OPENAI_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --kind OpenAI \
  --sku S0
```

Deploy a model (e.g., gpt-5.2):

```bash
az cognitiveservices account deployment create \
  --name $OPENAI_NAME \
  --resource-group $RESOURCE_GROUP \
  --deployment-name "gpt-5.2" \
  --model-name "gpt-5.2" \
  --model-version "2024-07-18" \
  --model-format OpenAI \
  --sku-capacity 10 \
  --sku-name Standard
```

Get the endpoint and key:

```bash
# Endpoint
az cognitiveservices account show \
  --name $OPENAI_NAME \
  --resource-group $RESOURCE_GROUP \
  --query properties.endpoint -o tsv

# Key
az cognitiveservices account keys list \
  --name $OPENAI_NAME \
  --resource-group $RESOURCE_GROUP \
  --query key1 -o tsv
```

### 1.4 Create App Service Plan

```bash
APP_SERVICE_PLAN="moltbot-plan"

az appservice plan create \
  --name $APP_SERVICE_PLAN \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --is-linux \
  --sku B1
```

### 1.5 Create Web App

```bash
WEB_APP_NAME="moltbot-app"  # Must be globally unique

az webapp create \
  --resource-group $RESOURCE_GROUP \
  --plan $APP_SERVICE_PLAN \
  --name $WEB_APP_NAME \
  --deployment-container-image-name "${ACR_NAME}.azurecr.io/moltbot-azure:latest"
```

## Step 2: Configure the Dockerfile

Create `Dockerfile.azure` in the repository root:

```dockerfile
FROM node:22-bookworm

# Install bun for faster builds
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:${PATH}"
RUN corepack enable

WORKDIR /app

# Copy package files
COPY package.json pnpm-lock.yaml ./
COPY patches ./patches

# Install dependencies
RUN pnpm install --frozen-lockfile

# Copy source and build
COPY . .
RUN pnpm build

# Prune dev dependencies
RUN pnpm prune --prod

ENV NODE_ENV=production

# Create config directory and bake in Azure-specific config
# IMPORTANT: trustedProxies is required for Azure's internal proxy
RUN mkdir -p /home/node/.moltbot && \
  echo '{"gateway":{"controlUi":{"allowInsecureAuth":true},"trustedProxies":["169.254.0.0/16"]},"agents":{"defaults":{"model":{"primary":"azure-openai/gpt-5.2"}}}}' > /home/node/.moltbot/moltbot.json && \
    chown -R node:node /home/node/.moltbot

USER node

EXPOSE 18789

CMD ["node", "dist/index.js", "gateway", "--bind", "lan", "--port", "18789", "--allow-unconfigured"]
```

### Key Configuration Notes

| Setting | Purpose |
|---------|---------|
| `allowInsecureAuth: true` | Control UI uses token-only auth (skips device pairing). Security downgrade; prefer HTTPS + pairing when possible. |
| `trustedProxies: ["169.254.0.0/16"]` | Trust Azure's internal proxy for `X-Forwarded-*` headers |
| `--bind lan` | Bind to all interfaces (required for container networking) |
| `--allow-unconfigured` | Start without requiring pre-configured channels |

## Step 3: Build and Push Container Image

### 3.1 Build with ACR Tasks (Recommended)

```bash
az acr build \
  --registry $ACR_NAME \
  --image moltbot-azure:latest \
  --file Dockerfile.azure \
  .
```

### 3.2 Alternative: Build Locally and Push

```bash
# Login to ACR
az acr login --name $ACR_NAME

# Build
docker build -t ${ACR_NAME}.azurecr.io/moltbot-azure:latest -f Dockerfile.azure .

# Push
docker push ${ACR_NAME}.azurecr.io/moltbot-azure:latest
```

## Step 4: Configure Web App

### 4.1 Connect ACR to Web App

```bash
# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)

# Configure container settings
az webapp config container set \
  --resource-group $RESOURCE_GROUP \
  --name $WEB_APP_NAME \
  --docker-custom-image-name "${ACR_NAME}.azurecr.io/moltbot-azure:latest" \
  --docker-registry-server-url "https://${ACR_NAME}.azurecr.io" \
  --docker-registry-server-user $ACR_USERNAME \
  --docker-registry-server-password $ACR_PASSWORD
```

### 4.2 Configure Environment Variables

```bash
# Generate a secure gateway token
GATEWAY_TOKEN=$(openssl rand -hex 16)

az webapp config appsettings set \
  --resource-group $RESOURCE_GROUP \
  --name $WEB_APP_NAME \
  --settings \
    WEBSITES_PORT=18789 \
    MOLTBOT_STATE_DIR=/home/node/.moltbot \
    MOLTBOT_CONFIG_PATH=/home/node/.moltbot/moltbot.json \
    CLAWDBOT_GATEWAY_TOKEN=$GATEWAY_TOKEN \
    AZURE_OPENAI_API_KEY="<your-azure-openai-key>" \
    AZURE_OPENAI_DEPLOYMENT_NAME="gpt-5.2" \
    AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/" \
    AZURE_OPENAI_API_VERSION="2024-08-01-preview"

echo "Gateway Token: $GATEWAY_TOKEN"
```

### 4.3 Enable WebSockets and Always On

```bash
az webapp config set \
  --resource-group $RESOURCE_GROUP \
  --name $WEB_APP_NAME \
  --web-sockets-enabled true \
  --always-on true
```

### 4.4 Set Startup Command

```bash
az webapp config set \
  --resource-group $RESOURCE_GROUP \
  --name $WEB_APP_NAME \
  --startup-file 'node dist/index.js gateway --bind lan --port 18789 --allow-unconfigured'
```

## Step 5: Deploy and Verify

### 5.1 Restart the Web App

```bash
az webapp restart --resource-group $RESOURCE_GROUP --name $WEB_APP_NAME
```

### 5.2 Check Logs

```bash
# Stream logs
az webapp log tail --resource-group $RESOURCE_GROUP --name $WEB_APP_NAME

# Or download logs
az webapp log download \
  --resource-group $RESOURCE_GROUP \
  --name $WEB_APP_NAME \
  --log-file /tmp/webapp-logs.zip
```

### 5.3 Verify Deployment

Look for these indicators in the logs:

```
[gateway] agent model: azure-openai/gpt-5.2
[gateway] listening on 0.0.0.0:18789
```

If you see warnings about untrusted proxy headers, the `trustedProxies` config is not applied correctly.

### 5.4 Access Control UI

Open in browser:

```
https://<your-app-name>.azurewebsites.net/?token=<your-gateway-token>
```

## Troubleshooting

### Problem: "pairing required" Error

**Cause**: The gateway doesn't recognize the connection as authorized.

**Solutions**:
1. Ensure `dangerouslyDisableDeviceAuth: true` is in the config
2. Ensure `trustedProxies` includes `169.254.0.0/16`
3. Access with the token: `?token=<your-token>`

### Problem: "Proxy headers detected from untrusted address"

**Cause**: Azure's internal proxy IP is not trusted.

**Solution**: Add `"trustedProxies": ["169.254.0.0/16"]` to gateway config.

### Problem: Container Crash Loop

**Check logs for**:
- `Cannot find module` - Startup command is malformed
- `SyntaxError` - Config JSON is invalid
- Port binding errors - Ensure `WEBSITES_PORT=18789`

**Common fix**: Verify `appCommandLine` is set correctly:

```bash
az webapp config show -g $RESOURCE_GROUP -n $WEB_APP_NAME --query appCommandLine
```

If it shows `""` (empty quotes), the image CMD won't be used. Set it explicitly:

```bash
az webapp config set -g $RESOURCE_GROUP -n $WEB_APP_NAME \
  --startup-file 'node dist/index.js gateway --bind lan --port 18789 --allow-unconfigured'
```

### Problem: Azure OpenAI Not Working

**Verify**:
1. Model deployment name matches config (e.g., `gpt-5.2`)
2. API key is correct
3. Endpoint URL format: `https://<resource>.openai.azure.com/`
4. API version is supported (e.g., `2024-08-01-preview`)

### Problem: 503 Service Unavailable

**Causes**:
- Container still starting (wait 1-2 minutes)
- Container crashing (check logs)
- Port mismatch (verify `WEBSITES_PORT`)

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `WEBSITES_PORT` | Yes | Port the container listens on (18789) |
| `MOLTBOT_STATE_DIR` | Yes | Directory for runtime state |
| `MOLTBOT_CONFIG_PATH` | Yes | Path to config JSON file |
| `CLAWDBOT_GATEWAY_TOKEN` | Yes | Token for Control UI authentication |
| `AZURE_OPENAI_API_KEY` | Yes | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Yes | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_VERSION` | Yes | API version (e.g., 2024-08-01-preview) |

## Security Recommendations

1. **Use Managed Identity** instead of API keys where possible
2. **Restrict network access** with Azure Private Endpoints
3. **Enable HTTPS Only** in Web App settings
4. **Rotate gateway tokens** periodically
5. **Use Azure Key Vault** for secrets in production

## Updating the Deployment

To deploy a new version:

```bash
# Rebuild image
az acr build --registry $ACR_NAME --image moltbot-azure:latest --file Dockerfile.azure .

# Restart to pull new image
az webapp restart --resource-group $RESOURCE_GROUP --name $WEB_APP_NAME
```

## Cleanup

To remove all resources:

```bash
az group delete --name $RESOURCE_GROUP --yes --no-wait
```
