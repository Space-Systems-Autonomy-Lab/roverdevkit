"""Service layer: route handlers delegate to functions here.

Routes stay thin (parse request, call service, format response); all
business logic — feature-row construction, surrogate dispatch, future
NSGA-II orchestration — lives under this package.
"""
