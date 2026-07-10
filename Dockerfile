# Static-only deploy of the NeuroAD demo workbench for Cloud Run.
# ONLY the two self-contained front-end assets are copied into the image —
# no Python backend, no src/, no data/, no reports/, no secrets, no weights.
FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY app/index.html /usr/share/nginx/html/index.html
COPY app/demo_data.json /usr/share/nginx/html/demo_data.json

EXPOSE 8080
