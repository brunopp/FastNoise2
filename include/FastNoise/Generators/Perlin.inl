#include "Perlin.h"
#include "Utils.inl"

template<FastSIMD::FeatureSet SIMD>
class FastSIMD::DispatchClass<Perlin, SIMD> final : public virtual Perlin, public DispatchClass<VariableRange<Seeded<ScalableGenerator>>, SIMD>
{
    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y ) const
    {
        seed += int32v( mSeedOffset );
        this->ScalePositions( x, y );

        float32v xs = FS::Floor( x );
        float32v ys = FS::Floor( y );

        int32v x0 = FS::Convert<int32_t>( xs ) * int32v( Primes::X );
        int32v y0 = FS::Convert<int32_t>( ys ) * int32v( Primes::Y );
        int32v x1 = x0 + int32v( Primes::X );
        int32v y1 = y0 + int32v( Primes::Y );

        float32v xf0 = xs = x - xs;
        float32v yf0 = ys = y - ys;
        float32v xf1 = xf0 - float32v( 1 );
        float32v yf1 = yf0 - float32v( 1 );

        xs = InterpQuintic( xs );
        ys = InterpQuintic( ys );

        float32v value = Lerp(
            Lerp( GetGradientDotPerlin( HashPrimes( seed, x0, y0 ), xf0, yf0 ), GetGradientDotPerlin( HashPrimes( seed, x1, y0 ), xf1, yf0 ), xs ),
            Lerp( GetGradientDotPerlin( HashPrimes( seed, x0, y1 ), xf0, yf1 ), GetGradientDotPerlin( HashPrimes( seed, x1, y1 ), xf1, yf1 ), xs ), ys );

        constexpr float kBounding = 1.726796627044677734375f;

        return this->ScaleOutput( value, -kBounding, kBounding );
    }

    gradientv FS_VECTORCALL GenD( int32v seed, float32v x, float32v y ) const final
    {
        float32v xs = FS::Floor( x );
        float32v ys = FS::Floor( y );

        int32v x0 = FS::Convert<int32_t>( xs ) * int32v( Primes::X );
        int32v y0 = FS::Convert<int32_t>( ys ) * int32v( Primes::Y );
        int32v x1 = x0 + int32v( Primes::X );
        int32v y1 = y0 + int32v( Primes::Y );

        float32v xf0 = xs = x - xs;
        float32v yf0 = ys = y - ys;
        float32v xf1 = xf0 - float32v( 1 );
        float32v yf1 = yf0 - float32v( 1 );
        // xf0 dx = 1 	xf0 dy = 0
        // yf0 dx = 0 	yf0 dy = 1
        // xf1 dx = 1	xf1 dy = 0
        // yf1 dx = 0	yf1 dy = 1

        float32v xu = InterpQuintic( xs ); // xu = xs * xs * xs * (xs*(xs*6-15)+10)
        float32v yu = InterpQuintic( ys ); // yu = ys * ys * ys * (ys*(ys*6-15)+10)
        // xsdx = xs*xs*(xs*(3*xs-60)+30) = 30*xs*xs*(xs*(xs-2)+1)
        // xsdy = 0
        // ysdx = 0
        // ysdy = ys*ys*(ys*(3*ys-60)+30)
        float32v xudx = float32v( 30 ) * xs * xs * ( xs * ( xs - float32v( 2 ) ) + float32v( 1 ) );
        float32v xudy = float32v( 0 );
        float32v yudx = float32v( 0 );
        float32v yudy = float32v( 30 ) * ys * ys * ( ys * ( ys - float32v( 2 ) ) + float32v( 1 ) );

        int32v h0 = HashPrimes( seed, x0, y0 );
        int32v h1 = HashPrimes( seed, x1, y0 );
        int32v h2 = HashPrimes( seed, x0, y1 );
        int32v h3 = HashPrimes( seed, x1, y1 );

        float32v gX[4];
        float32v gY[4];
        float32v g0 = GetGradientDotPerlinC( h0, xf0, yf0, gX[0], gY[0] ); //= gX*fX + fY*gY     = gX * xf0 + yf0 * gY
        float32v g1 = GetGradientDotPerlinC( h1, xf1, yf0, gX[1], gY[1] ); // 					= gX * xf1 + yf0 * gY
        float32v g2 = GetGradientDotPerlinC( h2, xf0, yf1, gX[2], gY[2] ); // 					= gX * xf0 + yf1 * gY
        float32v g3 = GetGradientDotPerlinC( h3, xf1, yf1, gX[3], gY[3] ); // 					= gX * xf1 + yf1 * gY
        // g0dx 	= (gX*fX + fY*gY)'
        //		= (gX*fX)' + (fY*gY)'
        //		= (gX'*fX)+(gX*fX') + (fY'*gY)+(fY*gY')
        //		= (0*fX)+(gX*1) + (0*gY)+(fY*0)
        //		= gX
        //		= gX[0]
        // g0dy 	= (gX*fX + fY*gY)'
        //		= (gX*fX)' + (fY*gY)'
        //		= (gX'*fX)+(gX*fX') + (fY'*gY)+(fY*gY')
        //		= (0*fX)+(gX*0) + (1*gY)+(fY*0)
        //		= gY
        //		= gY[0]

        float32v g0dx = gX[0];
        float32v g0dy = gY[0];
        float32v g1dx = gX[1];
        float32v g1dy = gY[1];
        float32v g2dx = gX[2];
        float32v g2dy = gY[2];
        float32v g3dx = gX[3];
        float32v g3dy = gY[3];

        float32v c = float32v( 0.579106986522674560546875f );

        float32v v = c *
            ( g0 + xu * g1 - xu * g0 + yu * xu * g3 - yu * xu * g2 - yu * xu * g1 + yu * xu * g0 + yu * g2 - yu * g0 );

        float32v vdx = c *
            ( g0dx // + (g0)' = g0dx
              + xudx * g1 + xu * g1dx // + (xu*g1)' = (xu'*g1 + xu*g1') = (xudx*g1 + x*g1dx) = xudx*g1 + xu*g1dx
              - xudx * g0 - xu * g0dx + yu * xudx * g3 + yu * xu * g3dx // + (yu*xu*g3) = (yu'*xu*g3)+(yu*xu'*g3)+(yu*xu*g3') = (ysdx*xu*g3)+(yu*xudx*g3)+(yu*xu*g3dx) = (0*xu*g3)+(yu*xudx*g3)+(yu*xu*g3dx) = yu*xudx*g3 + yu*xu*g3dx
              - yu * xudx * g2 - yu * xu * g2dx - yu * xudx * g1 - yu * xu * g1dx + yu * xudx * g0 + yu * xu * g0dx + yu * g2dx // + (yu*g2)' = (yu'*g2)+(yu*g2') = (0*g2)+(yu*g2dx) = yu*g2dx
              - yu * g0dx );

        float32v vdy = c *
            ( g0dy // + (g0)' = g0dy
              + xu * g1dy // + (xu*g1)' = (xu'*g1)+(xu*g1') = (0*g1)+(xu*g1dy) = xu*g1dy
              - xu * g0dy + xu * yudy * g3 + yu * xu * g3dy // + (yu*xu*g3)' = (yu'*xu*g3)+(yu*xu'*g3)+(yu*xu*g3') = (yudy*xu*g3)+(yu*0*g3)+(yu*xu*g3dy) = (yudy*xu*g3)+(yu*xu*g3dy) = yudy*xu*g3 + yu*xu*g3dy
              - xu * yudy * g2 - yu * xu * g2dy - xu * yudy * g1 - yu * xu * g1dy + xu * yudy * g0 + yu * xu * g0dy + yudy * g2 + yu * g2dy // + (yu*g2)' = (yu'*g2)+(yu*g2') = (yudy*g2)+(yu*g2dy) = yudy*g2 + yu*g2dy
              - yudy * g0 - yu * g0dy );

        constexpr float kBounding = 1.726796627044677734375f;

        gradientv g( this->ScaleOutput( v, -kBounding, kBounding ), vdx, vdy, float32v( 0 ) );
        return g;
    }

    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y, float32v z ) const
    {
        seed += int32v( mSeedOffset );
        this->ScalePositions( x, y, z );

        float32v xs = FS::Floor( x );
        float32v ys = FS::Floor( y );
        float32v zs = FS::Floor( z );

        int32v x0 = FS::Convert<int32_t>( xs ) * int32v( Primes::X );
        int32v y0 = FS::Convert<int32_t>( ys ) * int32v( Primes::Y );
        int32v z0 = FS::Convert<int32_t>( zs ) * int32v( Primes::Z );
        int32v x1 = x0 + int32v( Primes::X );
        int32v y1 = y0 + int32v( Primes::Y );
        int32v z1 = z0 + int32v( Primes::Z );

        float32v xf0 = xs = x - xs;
        float32v yf0 = ys = y - ys;
        float32v zf0 = zs = z - zs;
        float32v xf1 = xf0 - float32v( 1 );
        float32v yf1 = yf0 - float32v( 1 );
        float32v zf1 = zf0 - float32v( 1 );

        xs = InterpQuintic( xs );
        ys = InterpQuintic( ys );
        zs = InterpQuintic( zs );

        float32v value = Lerp( Lerp(
            Lerp( GetGradientDotCommon( HashPrimes( seed, x0, y0, z0 ), xf0, yf0, zf0 ), GetGradientDotCommon( HashPrimes( seed, x1, y0, z0 ), xf1, yf0, zf0 ), xs ),
            Lerp( GetGradientDotCommon( HashPrimes( seed, x0, y1, z0 ), xf0, yf1, zf0 ), GetGradientDotCommon( HashPrimes( seed, x1, y1, z0 ), xf1, yf1, zf0 ), xs ), ys ),
            Lerp(
            Lerp( GetGradientDotCommon( HashPrimes( seed, x0, y0, z1 ), xf0, yf0, zf1 ), GetGradientDotCommon( HashPrimes( seed, x1, y0, z1 ), xf1, yf0, zf1 ), xs ),
            Lerp( GetGradientDotCommon( HashPrimes( seed, x0, y1, z1 ), xf0, yf1, zf1 ), GetGradientDotCommon( HashPrimes( seed, x1, y1, z1 ), xf1, yf1, zf1 ), xs ), ys ), zs );

        constexpr double kBounding = 1.0363423824310302734375;

        return this->ScaleOutput( value, -kBounding, kBounding );
    }

    float32v FS_VECTORCALL Gen( int32v seed, float32v x, float32v y, float32v z, float32v w ) const
    {
        seed += int32v( mSeedOffset );
        this->ScalePositions( x, y, z, w );

        float32v xs = FS::Floor( x );
        float32v ys = FS::Floor( y );
        float32v zs = FS::Floor( z );
        float32v ws = FS::Floor( w );

        int32v x0 = FS::Convert<int32_t>( xs ) * int32v( Primes::X );
        int32v y0 = FS::Convert<int32_t>( ys ) * int32v( Primes::Y );
        int32v z0 = FS::Convert<int32_t>( zs ) * int32v( Primes::Z );
        int32v w0 = FS::Convert<int32_t>( ws ) * int32v( Primes::W );
        int32v x1 = x0 + int32v( Primes::X );
        int32v y1 = y0 + int32v( Primes::Y );
        int32v z1 = z0 + int32v( Primes::Z );
        int32v w1 = w0 + int32v( Primes::W );

        float32v xf0 = xs = x - xs;
        float32v yf0 = ys = y - ys;
        float32v zf0 = zs = z - zs;
        float32v wf0 = ws = w - ws;
        float32v xf1 = xf0 - float32v( 1 );
        float32v yf1 = yf0 - float32v( 1 );
        float32v zf1 = zf0 - float32v( 1 );
        float32v wf1 = wf0 - float32v( 1 );

        xs = InterpQuintic( xs );
        ys = InterpQuintic( ys );
        zs = InterpQuintic( zs );
        ws = InterpQuintic( ws );

        float32v value = Lerp( Lerp( Lerp(
            Lerp( GetGradientDotPerlin( HashPrimes( seed, x0, y0, z0, w0 ), xf0, yf0, zf0, wf0 ), GetGradientDotPerlin( HashPrimes( seed, x1, y0, z0, w0 ), xf1, yf0, zf0, wf0 ), xs ),
            Lerp( GetGradientDotPerlin( HashPrimes( seed, x0, y1, z0, w0 ), xf0, yf1, zf0, wf0 ), GetGradientDotPerlin( HashPrimes( seed, x1, y1, z0, w0 ), xf1, yf1, zf0, wf0 ), xs ), ys ),
            Lerp(
            Lerp( GetGradientDotPerlin( HashPrimes( seed, x0, y0, z1, w0 ), xf0, yf0, zf1, wf0 ), GetGradientDotPerlin( HashPrimes( seed, x1, y0, z1, w0 ), xf1, yf0, zf1, wf0 ), xs ),
            Lerp( GetGradientDotPerlin( HashPrimes( seed, x0, y1, z1, w0 ), xf0, yf1, zf1, wf0 ), GetGradientDotPerlin( HashPrimes( seed, x1, y1, z1, w0 ), xf1, yf1, zf1, wf0 ), xs ), ys ), zs ),
            Lerp( Lerp(
            Lerp( GetGradientDotPerlin( HashPrimes( seed, x0, y0, z0, w1 ), xf0, yf0, zf0, wf1 ), GetGradientDotPerlin( HashPrimes( seed, x1, y0, z0, w1 ), xf1, yf0, zf0, wf1 ), xs ),
            Lerp( GetGradientDotPerlin( HashPrimes( seed, x0, y1, z0, w1 ), xf0, yf1, zf0, wf1 ), GetGradientDotPerlin( HashPrimes( seed, x1, y1, z0, w1 ), xf1, yf1, zf0, wf1 ), xs ), ys ),
            Lerp(
            Lerp( GetGradientDotPerlin( HashPrimes( seed, x0, y0, z1, w1 ), xf0, yf0, zf1, wf1 ), GetGradientDotPerlin( HashPrimes( seed, x1, y0, z1, w1 ), xf1, yf0, zf1, wf1 ), xs ),
            Lerp( GetGradientDotPerlin( HashPrimes( seed, x0, y1, z1, w1 ), xf0, yf1, zf1, wf1 ), GetGradientDotPerlin( HashPrimes( seed, x1, y1, z1, w1 ), xf1, yf1, zf1, wf1 ), xs ), ys ) , zs ), ws );

        constexpr float kBounding = 1.33858621120452880859375;

        return this->ScaleOutput( value, -kBounding, kBounding );
    }
};
