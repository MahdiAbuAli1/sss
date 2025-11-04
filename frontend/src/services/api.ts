import type { ApiResponse } from '@/types'

// إعداد عنوان API الأساسي
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3001/api'

// فئة لخدمات HTTP
export class ApiService {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  // إرسال طلب GET
  async get<T>(endpoint: string): Promise<ApiResponse<T>> {
    const response = await fetch(`${this.baseUrl}${endpoint}`)
    return this.handleResponse(response)
  }

  // إرسال طلب POST
  async post<T>(endpoint: string, data?: any): Promise<ApiResponse<T>> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: data ? JSON.stringify(data) : undefined,
    })
    return this.handleResponse(response)
  }

  // إرسال طلب PUT
  async put<T>(endpoint: string, data?: any): Promise<ApiResponse<T>> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: data ? JSON.stringify(data) : undefined,
    })
    return this.handleResponse(response)
  }

  // إرسال طلب DELETE
  async delete<T>(endpoint: string): Promise<ApiResponse<T>> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'DELETE',
    })
    return this.handleResponse(response)
  }

  // معالجة الاستجابة
  private async handleResponse<T>(response: Response): Promise<ApiResponse<T>> {
    const data = await response.json()
    
    if (!response.ok) {
      throw new Error(data.message || 'حدث خطأ في الطلب')
    }

    return data
  }
}

// تصدير مثيل افتراضي من ApiService
export const apiService = new ApiService()