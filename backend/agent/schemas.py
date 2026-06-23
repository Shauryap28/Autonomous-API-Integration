"""
ApiSchema — the validated contract Phase 1 produces and every later phase reads.

Design choices:
  • Enums for auth_method / pagination.type -> a clean, machine-usable contract
    for the codegen step (no free-text the next phase has to interpret).
  • Headers as a list[Header], NOT an open dict -> controlled generation handles
    fixed-property objects and arrays far more reliably than arbitrary maps.
  • No Optional fields -> defaults instead, to avoid null-handling edge cases in
    structured output.
"""
from enum import Enum
from pydantic import BaseModel, Field


class AuthMethod(str, Enum):
    none = "none"
    api_key = "api_key"
    bearer = "bearer"
    oauth2 = "oauth2"


class PaginationType(str, Enum):
    none = "none"
    page = "page"        # ?page= / ?per_page=
    offset = "offset"    # ?offset= / ?limit=
    cursor = "cursor"    # opaque cursor / Link header


class Header(BaseModel):
    name: str
    value: str = Field(default="", description="Required/recommended value, if the docs state one.")


class Parameter(BaseModel):
    name: str
    location: str = Field(default="query", description="where it goes: query | path | header")
    required: bool = False
    description: str = ""


class Pagination(BaseModel):
    type: PaginationType = PaginationType.none
    param_names: list[str] = Field(default_factory=list, description="e.g. ['page', 'per_page']")
    notes: str = Field(default="", description="e.g. 'next page via the Link header'")


class ApiSchema(BaseModel):
    auth_method: AuthMethod
    base_url: str = Field(description="e.g. https://api.github.com")
    endpoint: str = Field(description="path template, e.g. /orgs/{org}/repos")
    http_method: str = "GET"
    required_headers: list[Header] = Field(default_factory=list)
    parameters: list[Parameter] = Field(default_factory=list)
    pagination: Pagination = Field(default_factory=Pagination)
    response_data_path: str = Field(
        default="",
        description="JSON path to the list of records; empty means the records are the top-level array.",
    )
    rate_limit: str = Field(default="", description="e.g. '60/hr unauthenticated, 5000/hr authenticated'")
    success_criteria: str = Field(default="", description="what a good response looks like, e.g. 'HTTP 200 + non-empty array'")
    notes: str = ""