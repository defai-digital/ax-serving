{{- define "ax-serving.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "ax-serving.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "ax-serving.labels" -}}
app.kubernetes.io/name: {{ include "ax-serving.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: gateway
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{- end -}}

{{- define "ax-serving.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ax-serving.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: gateway
{{- end -}}

{{- define "ax-serving.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "ax-serving.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{- define "ax-serving.image" -}}
{{- if .Values.image.digest -}}
{{- printf "%s@%s" .Values.image.repository .Values.image.digest -}}
{{- else if .Values.image.tag -}}
{{- printf "%s:%s" .Values.image.repository .Values.image.tag -}}
{{- else -}}
{{- printf "%s:%s" .Values.image.repository .Chart.AppVersion -}}
{{- end -}}
{{- end -}}

{{- define "ax-serving.validate" -}}
{{- $pub := int .Values.service.public.port -}}
{{- $ctl := int .Values.service.control.port -}}
{{- if eq $pub $ctl -}}
{{- fail "service.public.port and service.control.port must differ" -}}
{{- end -}}
{{- $prop := int .Values.gateway.shutdown.propagationMilliseconds -}}
{{- $drain := int .Values.gateway.shutdown.drainSeconds -}}
{{- $hard := int .Values.gateway.shutdown.hardSeconds -}}
{{- $combined := add $prop (mul $drain 1000) -}}
{{- if le (mul $hard 1000) $combined -}}
{{- fail "gateway.shutdown.hardSeconds must exceed propagationMilliseconds + drainSeconds" -}}
{{- end -}}
{{- if le (int .Values.gateway.terminationGracePeriodSeconds) $hard -}}
{{- fail "gateway.terminationGracePeriodSeconds must be greater than gateway.shutdown.hardSeconds" -}}
{{- end -}}
{{- if and (gt (int .Values.gateway.replicaCount) 1) (eq (default "memory" .Values.config.inline.orchestrator.fleet_store) "memory") (empty .Values.config.existingConfigMap) -}}
{{- fail "replicaCount > 1 requires orchestrator.fleet_store=redis (shared fleet state)" -}}
{{- end -}}
{{- if and .Values.production.enabled (empty .Values.secrets.existingSecret) -}}
{{- fail "production.enabled requires secrets.existingSecret" -}}
{{- end -}}
{{- if and .Values.production.enabled .Values.production.requireImageDigest (empty .Values.image.digest) -}}
{{- fail "production.enabled requires image.digest when requireImageDigest is true" -}}
{{- end -}}
{{- /* Reject GPU resources if an operator tries to inject them through resources maps. */ -}}
{{- with .Values.gateway.resources.limits -}}
{{- if hasKey . "nvidia.com/gpu" -}}
{{- fail "gateway resources must not request nvidia.com/gpu" -}}
{{- end -}}
{{- end -}}
{{- with .Values.gateway.resources.requests -}}
{{- if hasKey . "nvidia.com/gpu" -}}
{{- fail "gateway resources must not request nvidia.com/gpu" -}}
{{- end -}}
{{- end -}}
{{- end -}}
