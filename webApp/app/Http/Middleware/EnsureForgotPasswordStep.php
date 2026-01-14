<?php
namespace App\Http\Middleware;

use Closure;
use Illuminate\Http\Request;
use Symfony\Component\HttpFoundation\Response;

class EnsureForgotPasswordStep
{
    /**
     * Handle an incoming request.
     *
     * @param  \Closure(\Illuminate\Http\Request): (\Symfony\Component\HttpFoundation\Response)  $next
     */
    public function handle(Request $request, Closure $next, $step): Response
    {
        $steps = [
            'otp'    => 'email_verification_passed',
            'forgot' => 'otp_forgot_passed',
        ];

        if (! isset($steps[$step]) || ! session()->has($steps[$step])) {
            return redirect()->route('emailVerification');
        }

        return $next($request);
    }
}
