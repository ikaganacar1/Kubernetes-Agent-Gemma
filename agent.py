import gradio as gr
from unsloth import FastLanguageModel
import torch
import json
import yaml
import subprocess
import re
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import traceback
from threading import Thread
from transformers import TextIteratorStreamer

# ============== Model Configuration ==============
print("Loading Kubernetes Agent Model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/ika/yzlm/llm/Kubex_Lmm_Finetune/gemma3-kubernetes-0.0.0/checkpoint-8175/",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
DEVICE = next(model.parameters()).device

# ============== Tool Definitions ==============
class ToolType(Enum):
    KUBECTL = "kubectl"
    HELM = "helm"
    YAML = "yaml"
    DIAGNOSTIC = "diagnostic"

@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    tool_type: ToolType
    examples: List[str]

# Define available tools for Kubernetes operations
KUBERNETES_TOOLS = [
    Tool(
        name="kubectl_command",
        description="Execute kubectl commands to interact with Kubernetes cluster",
        parameters={
            "command": {"type": "string", "description": "Full kubectl command to execute"},
            "namespace": {"type": "string", "description": "Target namespace", "default": "default"},
            "dry_run": {"type": "boolean", "description": "Execute as dry-run", "default": False}
        },
        tool_type=ToolType.KUBECTL,
        examples=[
            "kubectl get pods -n production",
            "kubectl describe deployment nginx",
            "kubectl logs pod-name --tail=100"
        ]
    ),
    Tool(
        name="create_yaml",
        description="Generate Kubernetes YAML manifest files for resources",
        parameters={
            "resource_type": {"type": "string", "description": "Type of K8s resource (deployment, service, configmap, etc.)"},
            "resource_name": {"type": "string", "description": "Name of the resource"},
            "namespace": {"type": "string", "description": "Target namespace"},
            "spec": {"type": "object", "description": "Resource specification"}
        },
        tool_type=ToolType.YAML,
        examples=[
            "Create a deployment YAML for nginx with 3 replicas",
            "Generate a service YAML for exposing port 80"
        ]
    ),
    Tool(
        name="helm_operation",
        description="Execute Helm operations for managing charts and releases",
        parameters={
            "operation": {"type": "string", "description": "Helm operation (install, upgrade, rollback, uninstall, list)"},
            "release_name": {"type": "string", "description": "Name of the Helm release"},
            "chart": {"type": "string", "description": "Chart reference or path"},
            "namespace": {"type": "string", "description": "Target namespace", "default": "default"},
            "values": {"type": "object", "description": "Values to override", "optional": True}
        },
        tool_type=ToolType.HELM,
        examples=[
            "helm install nginx bitnami/nginx",
            "helm upgrade production-app ./my-chart --values prod-values.yaml",
            "helm rollback my-release 2"
        ]
    ),
    Tool(
        name="troubleshoot_pod",
        description="Diagnose and troubleshoot pod issues",
        parameters={
            "pod_name": {"type": "string", "description": "Name of the pod to troubleshoot"},
            "namespace": {"type": "string", "description": "Pod namespace", "default": "default"},
            "check_type": {"type": "string", "description": "Type of check (events, logs, describe, resources)"}
        },
        tool_type=ToolType.DIAGNOSTIC,
        examples=[
            "Troubleshoot why nginx-pod is in CrashLoopBackOff",
            "Check events for pod mysql-primary"
        ]
    ),
    Tool(
        name="scale_resource",
        description="Scale Kubernetes deployments or statefulsets",
        parameters={
            "resource_type": {"type": "string", "description": "Type of resource (deployment, statefulset)"},
            "resource_name": {"type": "string", "description": "Name of the resource"},
            "replicas": {"type": "integer", "description": "Number of replicas"},
            "namespace": {"type": "string", "description": "Target namespace", "default": "default"}
        },
        tool_type=ToolType.KUBECTL,
        examples=[
            "Scale nginx deployment to 5 replicas",
            "Scale down redis statefulset to 1"
        ]
    ),
    Tool(
        name="apply_manifest",
        description="Apply Kubernetes manifest from YAML",
        parameters={
            "yaml_content": {"type": "string", "description": "YAML manifest content"},
            "namespace": {"type": "string", "description": "Target namespace", "default": "default"},
            "validate": {"type": "boolean", "description": "Validate before applying", "default": True}
        },
        tool_type=ToolType.KUBECTL,
        examples=[
            "Apply the provided deployment YAML",
            "Create resources from manifest"
        ]
    )
]

# ============== System Prompt ==============
AGENTIC_SYSTEM_PROMPT = """You are an advanced Kubernetes Agent with tool-calling capabilities. Your role is to help users manage Kubernetes clusters by generating commands, creating configurations, and executing operations.

## Your Capabilities:
1. **Command Generation**: Create and execute kubectl commands
2. **YAML Creation**: Generate Kubernetes manifest files
3. **Helm Operations**: Manage Helm charts and releases
4. **Troubleshooting**: Diagnose and resolve Kubernetes issues
5. **Resource Management**: Scale, update, and manage K8s resources

## Available Tools:
{tools_description}

## Function Calling Instructions:
When you need to use a tool, respond with a function call in this EXACT format:

```tool_call
{{
  "tool": "tool_name",
  "parameters": {{
    "param1": "value1",
    "param2": "value2"
  }}
}}
```

After receiving tool output, analyze the result and provide a helpful response.

## Response Format:
1. **REASONING**: First, explain what you're going to do and why
2. **TOOL_CALL**: If needed, make the appropriate tool call
3. **ANALYSIS**: After tool execution, explain the results
4. **RECOMMENDATIONS**: Suggest next steps or best practices

## Important Guidelines:
- Always validate user requests before executing commands
- Use dry-run when appropriate for safety
- Provide clear explanations of what each command does
- Follow Kubernetes best practices
- Include error handling and fallback options
- Be security-conscious (avoid exposing secrets, use RBAC properly)

## Examples of Tool Usage:

### Example 1: Creating a Deployment
User: "Create a nginx deployment with 3 replicas"
Response: I'll create a Kubernetes deployment for nginx with 3 replicas.

```tool_call
{{
  "tool": "create_yaml",
  "parameters": {{
    "resource_type": "deployment",
    "resource_name": "nginx-deployment",
    "namespace": "default",
    "spec": {{
      "replicas": 3,
      "image": "nginx:latest",
      "ports": [80]
    }}
  }}
}}
```

### Example 2: Troubleshooting
User: "My pod is in CrashLoopBackOff state"
Response: I'll help you troubleshoot the pod issue.

```tool_call
{{
  "tool": "troubleshoot_pod",
  "parameters": {{
    "pod_name": "problematic-pod",
    "namespace": "default",
    "check_type": "events"
  }}
}}
```

Remember: You are a production-grade Kubernetes agent. Prioritize reliability, security, and best practices in all operations."""

def format_tools_description(tools: List[Tool]) -> str:
    """Format tools for inclusion in system prompt"""
    descriptions = []
    for tool in tools:
        params_str = json.dumps(tool.parameters, indent=2)
        examples_str = "\n".join(f"  - {ex}" for ex in tool.examples[:2])
        descriptions.append(f"""
**{tool.name}** ({tool.tool_type.value})
Description: {tool.description}
Parameters: {params_str}
Examples:
{examples_str}
""")
    return "\n".join(descriptions)

# ============== Tool Execution Engine ==============
class ToolExecutor:
    """Execute tool calls and return results"""
    
    def __init__(self, dry_run_mode: bool = False):
        self.dry_run_mode = dry_run_mode
        self.execution_history = []
        
    def execute(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call and return the result"""
        try:
            tool_name = tool_call.get("tool")
            parameters = tool_call.get("parameters", {})
            
            # Log execution
            execution = {
                "timestamp": datetime.now().isoformat(),
                "tool": tool_name,
                "parameters": parameters,
                "status": "pending"
            }
            
            # Execute based on tool type
            if tool_name == "kubectl_command":
                result = self._execute_kubectl(parameters)
            elif tool_name == "create_yaml":
                result = self._create_yaml(parameters)
            elif tool_name == "helm_operation":
                result = self._execute_helm(parameters)
            elif tool_name == "troubleshoot_pod":
                result = self._troubleshoot_pod(parameters)
            elif tool_name == "scale_resource":
                result = self._scale_resource(parameters)
            elif tool_name == "apply_manifest":
                result = self._apply_manifest(parameters)
            else:
                result = {"error": f"Unknown tool: {tool_name}"}
            
            execution["status"] = "success" if "error" not in result else "failed"
            execution["result"] = result
            self.execution_history.append(execution)
            
            return result
            
        except Exception as e:
            error_result = {"error": str(e), "traceback": traceback.format_exc()}
            execution["status"] = "error"
            execution["result"] = error_result
            self.execution_history.append(execution)
            return error_result
    
    def _execute_kubectl(self, params: Dict) -> Dict:
        """Execute kubectl command"""
        command = params.get("command", "")
        namespace = params.get("namespace", "default")
        dry_run = params.get("dry_run", False)
        
        if dry_run or self.dry_run_mode:
            return {
                "output": f"[DRY-RUN] Would execute: {command}",
                "command": command,
                "dry_run": True
            }
        
        # In production, actually execute the command
        # For testing, return simulated output
        return {
            "output": f"Simulated output for: {command}",
            "command": command,
            "namespace": namespace
        }
    
    def _create_yaml(self, params: Dict) -> Dict:
        """Generate Kubernetes YAML"""
        resource_type = params.get("resource_type")
        resource_name = params.get("resource_name")
        namespace = params.get("namespace", "default")
        spec = params.get("spec", {})
        
        # Generate YAML based on resource type
        if resource_type == "deployment":
            yaml_content = self._generate_deployment_yaml(resource_name, namespace, spec)
        elif resource_type == "service":
            yaml_content = self._generate_service_yaml(resource_name, namespace, spec)
        elif resource_type == "configmap":
            yaml_content = self._generate_configmap_yaml(resource_name, namespace, spec)
        else:
            yaml_content = f"# Generated {resource_type} YAML\n# Implement specific generator"
        
        return {
            "yaml": yaml_content,
            "resource_type": resource_type,
            "resource_name": resource_name
        }
    
    def _generate_deployment_yaml(self, name: str, namespace: str, spec: Dict) -> str:
        """Generate deployment YAML"""
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {"app": name}
            },
            "spec": {
                "replicas": spec.get("replicas", 1),
                "selector": {"matchLabels": {"app": name}},
                "template": {
                    "metadata": {"labels": {"app": name}},
                    "spec": {
                        "containers": [{
                            "name": name,
                            "image": spec.get("image", "nginx:latest"),
                            "ports": [{"containerPort": p} for p in spec.get("ports", [80])]
                        }]
                    }
                }
            }
        }
        return yaml.dump(deployment, default_flow_style=False)
    
    def _generate_service_yaml(self, name: str, namespace: str, spec: Dict) -> str:
        """Generate service YAML"""
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": name,
                "namespace": namespace
            },
            "spec": {
                "selector": {"app": spec.get("selector", name)},
                "ports": [{
                    "port": spec.get("port", 80),
                    "targetPort": spec.get("targetPort", 80)
                }],
                "type": spec.get("type", "ClusterIP")
            }
        }
        return yaml.dump(service, default_flow_style=False)
    
    def _generate_configmap_yaml(self, name: str, namespace: str, spec: Dict) -> str:
        """Generate configmap YAML"""
        configmap = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": name,
                "namespace": namespace
            },
            "data": spec.get("data", {})
        }
        return yaml.dump(configmap, default_flow_style=False)
    
    def _execute_helm(self, params: Dict) -> Dict:
        """Execute Helm operation"""
        operation = params.get("operation")
        release_name = params.get("release_name")
        chart = params.get("chart", "")
        namespace = params.get("namespace", "default")
        values = params.get("values", {})
        
        if self.dry_run_mode:
            return {
                "output": f"[DRY-RUN] helm {operation} {release_name} {chart}",
                "operation": operation,
                "dry_run": True
            }
        
        # Simulate Helm operations
        outputs = {
            "install": f"Release {release_name} installed successfully",
            "upgrade": f"Release {release_name} upgraded to latest version",
            "rollback": f"Release {release_name} rolled back",
            "uninstall": f"Release {release_name} uninstalled",
            "list": f"Listing releases in namespace {namespace}"
        }
        
        return {
            "output": outputs.get(operation, f"Executed: helm {operation}"),
            "release": release_name,
            "namespace": namespace
        }
    
    def _troubleshoot_pod(self, params: Dict) -> Dict:
        """Troubleshoot pod issues"""
        pod_name = params.get("pod_name")
        namespace = params.get("namespace", "default")
        check_type = params.get("check_type", "describe")
        
        # Simulate troubleshooting checks
        checks = {
            "events": f"Events for pod {pod_name}:\n- Warning: BackOff - Back-off restarting failed container",
            "logs": f"Logs for pod {pod_name}:\nError: Cannot connect to database",
            "describe": f"Pod {pod_name} details:\nStatus: CrashLoopBackOff\nRestarts: 5",
            "resources": f"Resources for pod {pod_name}:\nCPU: 100m/500m\nMemory: 256Mi/1Gi"
        }
        
        return {
            "pod": pod_name,
            "namespace": namespace,
            "check_type": check_type,
            "diagnosis": checks.get(check_type, "Check completed"),
            "recommendations": [
                "Check container logs for errors",
                "Verify resource limits",
                "Check liveness/readiness probes"
            ]
        }
    
    def _scale_resource(self, params: Dict) -> Dict:
        """Scale Kubernetes resource"""
        resource_type = params.get("resource_type")
        resource_name = params.get("resource_name")
        replicas = params.get("replicas")
        namespace = params.get("namespace", "default")
        
        command = f"kubectl scale {resource_type} {resource_name} --replicas={replicas} -n {namespace}"
        
        if self.dry_run_mode:
            return {
                "output": f"[DRY-RUN] Would scale {resource_type}/{resource_name} to {replicas} replicas",
                "command": command,
                "dry_run": True
            }
        
        return {
            "output": f"Scaled {resource_type}/{resource_name} to {replicas} replicas",
            "resource": f"{resource_type}/{resource_name}",
            "replicas": replicas,
            "namespace": namespace
        }
    
    def _apply_manifest(self, params: Dict) -> Dict:
        """Apply Kubernetes manifest"""
        yaml_content = params.get("yaml_content")
        namespace = params.get("namespace", "default")
        validate = params.get("validate", True)
        
        if validate:
            # Validate YAML structure
            try:
                yaml.safe_load(yaml_content)
            except yaml.YAMLError as e:
                return {"error": f"Invalid YAML: {e}"}
        
        if self.dry_run_mode:
            return {
                "output": "[DRY-RUN] Would apply manifest",
                "validated": validate,
                "dry_run": True
            }
        
        return {
            "output": "Manifest applied successfully",
            "namespace": namespace,
            "validated": validate
        }

# ============== Response Parser ==============
class ResponseParser:
    """Parse model responses to extract tool calls"""
    
    @staticmethod
    def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
        """Extract tool calls from model response"""
        tool_calls = []
        
        # Pattern 1: ```tool_call JSON ```
        pattern1 = r'```tool_call\s*(.*?)\s*```'
        matches = re.findall(pattern1, text, re.DOTALL)
        
        for match in matches:
            try:
                tool_call = json.loads(match)
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue
        
        # Pattern 2: ```tool_code Python code ```
        pattern2 = r'```tool_code\s*(.*?)\s*```'
        matches = re.findall(pattern2, text, re.DOTALL)
        
        for match in matches:
            # Parse Python-style function calls
            tool_calls.append({"code": match, "type": "python"})
        
        return tool_calls
    
    @staticmethod
    def clean_response(text: str) -> str:
        """Remove tool call blocks from response text"""
        # Remove tool_call blocks
        text = re.sub(r'```tool_call.*?```', '', text, flags=re.DOTALL)
        # Remove tool_code blocks
        text = re.sub(r'```tool_code.*?```', '', text, flags=re.DOTALL)
        return text.strip()

# ============== Test Scenarios ==============
TEST_SCENARIOS = [
    {
        "name": "Deploy WordPress",
        "prompt": "Deploy a WordPress application with MySQL database using Helm",
        "expected_tools": ["helm_operation"],
        "validation": lambda result: "install" in str(result).lower()
    },
    {
        "name": "Create Nginx Deployment",
        "prompt": "Create a deployment YAML for nginx with 3 replicas exposed on port 80",
        "expected_tools": ["create_yaml"],
        "validation": lambda result: "deployment" in str(result).lower() and "replicas: 3" in str(result)
    },
    {
        "name": "Troubleshoot CrashLoopBackOff",
        "prompt": "My pod 'api-server' is in CrashLoopBackOff state, help me troubleshoot",
        "expected_tools": ["troubleshoot_pod", "kubectl_command"],
        "validation": lambda result: "crashloopbackoff" in str(result).lower()
    },
    {
        "name": "Scale Application",
        "prompt": "Scale the 'frontend' deployment to 5 replicas in production namespace",
        "expected_tools": ["scale_resource"],
        "validation": lambda result: "scale" in str(result).lower() and "5" in str(result)
    },
    {
        "name": "Create Service",
        "prompt": "Create a LoadBalancer service to expose my nginx deployment externally",
        "expected_tools": ["create_yaml"],
        "validation": lambda result: "service" in str(result).lower() and "loadbalancer" in str(result).lower()
    },
    {
        "name": "Helm Upgrade",
        "prompt": "Upgrade my 'monitoring' Helm release with the latest Prometheus chart",
        "expected_tools": ["helm_operation"],
        "validation": lambda result: "upgrade" in str(result).lower()
    },
    {
        "name": "Debug Pod Logs",
        "prompt": "Show me the last 50 lines of logs from the 'database' pod",
        "expected_tools": ["kubectl_command"],
        "validation": lambda result: "logs" in str(result).lower() and "tail" in str(result).lower()
    },
    {
        "name": "Create ConfigMap",
        "prompt": "Create a ConfigMap named 'app-config' with database connection settings",
        "expected_tools": ["create_yaml"],
        "validation": lambda result: "configmap" in str(result).lower()
    },
    {
        "name": "List Resources",
        "prompt": "List all deployments in the 'production' namespace with their replica count",
        "expected_tools": ["kubectl_command"],
        "validation": lambda result: "get deployment" in str(result).lower()
    },
    {
        "name": "Rollback Deployment",
        "prompt": "Rollback the 'api' deployment to the previous version",
        "expected_tools": ["kubectl_command", "helm_operation"],
        "validation": lambda result: "rollback" in str(result).lower() or "rollout undo" in str(result).lower()
    }
]

# ============== Evaluation Metrics ==============
class AgentEvaluator:
    """Evaluate agent performance on test scenarios"""
    
    def __init__(self):
        self.results = []
        
    def evaluate_response(self, scenario: Dict, response: str, tool_calls: List[Dict]) -> Dict:
        """Evaluate a single response"""
        result = {
            "scenario": scenario["name"],
            "prompt": scenario["prompt"],
            "response": response,
            "tool_calls": tool_calls,
            "metrics": {}
        }
        
        # Check if expected tools were called
        called_tools = [tc.get("tool") for tc in tool_calls]
        expected_tools = scenario.get("expected_tools", [])
        
        result["metrics"]["tool_accuracy"] = len(set(called_tools) & set(expected_tools)) / max(len(expected_tools), 1)
        
        # Check if validation passes
        validation_fn = scenario.get("validation", lambda x: True)
        result["metrics"]["validation_passed"] = validation_fn(response)
        
        # Check response quality
        result["metrics"]["has_reasoning"] = "will" in response.lower() or "going to" in response.lower()
        result["metrics"]["has_tool_calls"] = len(tool_calls) > 0
        
        # Calculate overall score
        result["metrics"]["score"] = (
            result["metrics"]["tool_accuracy"] * 0.4 +
            result["metrics"]["validation_passed"] * 0.3 +
            result["metrics"]["has_reasoning"] * 0.2 +
            result["metrics"]["has_tool_calls"] * 0.1
        )
        
        self.results.append(result)
        return result
    
    def generate_report(self) -> str:
        """Generate evaluation report"""
        if not self.results:
            return "No evaluation results available"
        
        report = "## Agent Evaluation Report\n\n"
        
        # Overall metrics
        avg_score = sum(r["metrics"]["score"] for r in self.results) / len(self.results)
        tool_accuracy = sum(r["metrics"]["tool_accuracy"] for r in self.results) / len(self.results)
        validation_rate = sum(r["metrics"]["validation_passed"] for r in self.results) / len(self.results)
        
        report += f"### Overall Performance\n"
        report += f"- Average Score: {avg_score:.2%}\n"
        report += f"- Tool Selection Accuracy: {tool_accuracy:.2%}\n"
        report += f"- Validation Pass Rate: {validation_rate:.2%}\n\n"
        
        # Per-scenario results
        report += "### Scenario Results\n"
        for result in self.results:
            report += f"\n**{result['scenario']}**\n"
            report += f"- Score: {result['metrics']['score']:.2%}\n"
            report += f"- Tool Accuracy: {result['metrics']['tool_accuracy']:.2%}\n"
            report += f"- Validation: {'✓' if result['metrics']['validation_passed'] else '✗'}\n"
            report += f"- Tools Called: {', '.join([tc.get('tool', 'unknown') for tc in result['tool_calls']])}\n"
        
        return report

# ============== Main Application ==============
class KubernetesAgentTester:
    """Main application for testing the Kubernetes agent"""
    
    def __init__(self):
        self.executor = ToolExecutor(dry_run_mode=True)
        self.parser = ResponseParser()
        self.evaluator = AgentEvaluator()
        self.system_prompt = AGENTIC_SYSTEM_PROMPT.format(
            tools_description=format_tools_description(KUBERNETES_TOOLS)
        )
        
    def process_message(self, message: str, history: List) -> str:
        """Process a message with tool calling support"""
        # Build conversation with agentic system prompt
        messages = self.build_conversation(message, history)
        
        # Generate response
        response = ""
        for chunk in self.generate_response_streaming(messages):
            response = chunk
            yield chunk
        
        # Extract and execute tool calls
        tool_calls = self.parser.extract_tool_calls(response)
        
        if tool_calls:
            yield "\n\n**Tool Execution Results:**\n"
            for tool_call in tool_calls:
                if tool_call.get("type") == "python":
                    result = {"output": "Python code execution not implemented in test mode"}
                else:
                    result = self.executor.execute(tool_call)
                
                yield f"\n```json\n{json.dumps(result, indent=2)}\n```\n"
                
                # Generate follow-up response based on tool output
                follow_up = f"\n\nBased on the tool execution:\n{self.analyze_tool_result(result)}"
                yield follow_up
    
    def build_conversation(self, message: str, history: List) -> List[Dict]:
        """Build conversation with agentic system prompt"""
        messages = []
        
        if not history:
            # First message with system prompt
            messages.append({
                "role": "user",
                "content": f"{self.system_prompt}\n\nUser Request: {message}"
            })
        else:
            # Add history
            for user_msg, assistant_msg in history:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "model", "content": assistant_msg})
            
            # Add current message
            messages.append({"role": "user", "content": message})
        
        return messages
    
    def generate_response_streaming(self, messages: List[Dict]) -> str:
        """Generate response with streaming"""
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=30.0
            )
            
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                do_sample=True,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )
            
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                if "<end_of_turn>" in generated_text:
                    generated_text = generated_text.replace("<end_of_turn>", "")
                    break
                yield generated_text
            
            thread.join()
            
            # Cleanup
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def analyze_tool_result(self, result: Dict) -> str:
        """Analyze tool execution result"""
        if "error" in result:
            return f"⚠️ The operation encountered an error: {result['error']}"
        elif result.get("dry_run"):
            return "✓ Command validated successfully (dry-run mode)"
        elif "yaml" in result:
            return "✓ YAML manifest generated successfully"
        elif "output" in result:
            return f"✓ Operation completed: {result['output']}"
        else:
            return "✓ Tool executed successfully"
    
    def run_test_suite(self) -> str:
        """Run all test scenarios"""
        results = []
        for scenario in TEST_SCENARIOS:
            print(f"Testing: {scenario['name']}")
            
            # Generate response
            response = ""
            for chunk in self.process_message(scenario["prompt"], []):
                response = chunk
            
            # Extract tool calls
            tool_calls = self.parser.extract_tool_calls(response)
            
            # Evaluate
            result = self.evaluator.evaluate_response(scenario, response, tool_calls)
            results.append(result)
        
        return self.evaluator.generate_report()

# ============== Gradio Interface ==============
def create_interface():
    tester = KubernetesAgentTester()
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚢 Kubernetes Agent Testing Framework")
        gr.Markdown("Test your fine-tuned Gemma 3 model's agentic capabilities for Kubernetes operations")
        
        with gr.Tab("Interactive Chat"):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(
                label="Enter your Kubernetes request",
                placeholder="e.g., 'Deploy nginx with 3 replicas' or 'Troubleshoot my CrashLoopBackOff pod'"
            )
            clear = gr.ClearButton([msg, chatbot])
            
            def respond(message, chat_history):
                bot_message = ""
                chat_history.append([message, ""])
                
                for chunk in tester.process_message(message, chat_history[:-1]):
                    bot_message = chunk
                    chat_history[-1][1] = bot_message
                    yield chat_history
            
            msg.submit(respond, [msg, chatbot], [chatbot])
        
        with gr.Tab("Test Scenarios"):
            gr.Markdown("### Predefined Test Scenarios")
            
            test_output = gr.Textbox(
                label="Test Results",
                lines=20,
                max_lines=30
            )
            
            run_tests = gr.Button("Run Test Suite", variant="primary")
            
            def run_suite():
                return tester.run_test_suite()
            
            run_tests.click(run_suite, outputs=[test_output])
            
            # Display available scenarios
            scenarios_md = "#### Available Test Scenarios:\n"
            for i, scenario in enumerate(TEST_SCENARIOS, 1):
                scenarios_md += f"{i}. **{scenario['name']}**: {scenario['prompt']}\n"
            gr.Markdown(scenarios_md)
        
        with gr.Tab("Tool Execution History"):
            history_output = gr.JSON(label="Execution History")
            refresh_history = gr.Button("Refresh History")
            
            def get_history():
                return tester.executor.execution_history
            
            refresh_history.click(get_history, outputs=[history_output])
        
        with gr.Tab("Examples"):
            gr.Examples(
                examples=[
                    "Create a production-ready nginx deployment with 5 replicas",
                    "Deploy PostgreSQL with persistent storage using Helm",
                    "My application pod is in ImagePullBackOff state, help me fix it",
                    "Create a horizontal pod autoscaler for my web-app deployment",
                    "Generate a service mesh configuration for Istio",
                    "Set up a CI/CD pipeline deployment in Kubernetes",
                    "Create a StatefulSet for Redis with master-slave configuration",
                    "Configure RBAC for a new developer with limited permissions",
                    "Set up monitoring with Prometheus and Grafana using Helm",
                    "Create an Ingress controller to expose my services externally"
                ],
                inputs=msg
            )
    
    return demo

if __name__ == "__main__":
    print("Starting Kubernetes Agent Testing Framework...")
    print(f"Model: Gemma 3 (Fine-tuned for Kubernetes)")
    print(f"Mode: Agentic with Tool Calling")
    print(f"Tools Available: {len(KUBERNETES_TOOLS)}")
    
    interface = create_interface()
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )