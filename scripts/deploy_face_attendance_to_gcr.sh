#!/usr/bin/env bash

set -e
set -o pipefail
set -u

echo "############# Deploying the Cloud Run service $SERVICE_NAME"

gcloud run deploy "$SERVICE_NAME" \
  --image "$LOCATION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$SERVICE_NAME:$IMAGE_TAG" \
  --region="$LOCATION" \
  --allow-unauthenticated \
  --set-env-vars PROJECT_ID="$PROJECT_ID" \
  --set-env-vars SUPABASE_URL="$SUPABASE_URL" \
  --set-env-vars SUPABASE_KEY="$SUPABASE_KEY" \
  --set-env-vars JWT_SECRET="$JWT_SECRET" \
  --set-env-vars JWT_ALGORITHM="$JWT_ALGORITHM" \