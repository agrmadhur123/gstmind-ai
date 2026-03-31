# GSTMind Backend API

## Environment Variables (set in Railway dashboard)
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
CLAUDE_API_KEY=sk-ant-api03-...
RAZORPAY_KEY_ID=rzp_live_...
RAZORPAY_KEY_SECRET=your-razorpay-secret
MSG91_AUTH_KEY=your-msg91-key
MSG91_TEMPLATE_ID=your-template-id
JWT_SECRET=your-random-secret-string
ADMIN_SECRET=your-admin-api-secret
```

## Endpoints
- POST /auth/signup/email
- POST /auth/login/email
- POST /auth/otp/send
- POST /auth/otp/verify
- GET  /auth/me
- POST /query  (protected)
- POST /payment/create-order (protected)
- POST /payment/verify (protected)
- GET  /admin/users (admin only)
- PATCH /admin/users/:id (admin only)
- GET  /admin/analytics (admin only)
- GET  /health

## Deploy
1. Push to GitHub
2. Connect repo on railway.app
3. Set environment variables
4. Deploy
