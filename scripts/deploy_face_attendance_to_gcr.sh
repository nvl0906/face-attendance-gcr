#!/usr/bin/env bash

set -e
set -o pipefail
set -u

echo "############# Deploying the Cloud Run service $SERVICE_NAME"

gcloud run deploy "$SERVICE_NAME" \
  --image "$LOCATION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$SERVICE_NAME:$IMAGE_TAG" \
  --region "$LOCATION" \
  --project "$PROJECT_ID" \
  --platform "managed" \
  --allow-unauthenticated \
  --set-secrets=SUPABASE_URL=SUPABASE_URL:latest,SUPABASE_KEY=SUPABASE_KEY:latest,JWT_SECRET=JWT_SECRET:latest,JWT_ALGORITHM=JWT_ALGORITHM:latest
  
