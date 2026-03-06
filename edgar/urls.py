from django.urls import path

from edgar import views

urlpatterns = [
    path("companies/", views.companies, name="edgar-companies"),
    path("companies/search/", views.company_search, name="edgar-company-search"),
    path("companies/<str:ticker>/documents/", views.company_documents, name="edgar-company-documents"),
    path("filings/fulltext/", views.fulltext_search, name="edgar-fulltext-search"),
]
