# Kubernetes Deployment

## Prerequisites

- `kubectl` configured to point at your cluster
- `kustomize` (built into `kubectl` v1.14+)
- Docker image `ai-platform-api:latest` built and accessible to the cluster

## Build the image

```bash
docker build -t ai-platform-api:latest .
```

## Deploy to dev

```bash
kubectl apply -k k8s/overlays/dev/
```

## Deploy to staging

```bash
kubectl apply -k k8s/overlays/staging/
```

## Verify rollout

```bash
kubectl rollout status deployment/ai-platform-api -n ai-platform
kubectl get pods -n ai-platform
```

## Update the API key secret (before first deploy)

```bash
kubectl create secret generic ai-platform-secrets \
  --from-literal=AIP_KAIROS_API_KEY=<your-key> \
  --namespace ai-platform \
  --dry-run=client -o yaml | kubectl apply -f -
```

## Check logs

```bash
kubectl logs -n ai-platform -l app=ai-platform-api --follow
```
