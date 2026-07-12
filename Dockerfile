# Static deploy of the REAL-DATA NeuroAD demo (neuroad.html + assets) for Cloud Run.
# Front-end assets only — NO Python backend, NO gated data/weights/secrets.
# Live /api/* (dynamic Investigate, SFG QC) is not served here; the real-data
# frozen view (tree, cards, brain viz, heatmap, blue crosshair) renders from the
# committed demo_data.json + public MNI template + demo ROI masks.
FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY app/neuroad.html /usr/share/nginx/html/index.html
COPY app/demo_data.json /usr/share/nginx/html/demo_data.json
COPY app/vendor/niivue.umd.js /usr/share/nginx/html/vendor/niivue.umd.js
COPY app/scans/ /usr/share/nginx/html/scans/

EXPOSE 8080
